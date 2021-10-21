"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import copy
from collections import OrderedDict
import yaml
from nncf.experimental.intel_tensorflow.utils.utility import LazyImport, CpuInfo, singleton
from nncf.experimental.intel_tensorflow.utils import logger

tensorflow = LazyImport('tensorflow')


class IntelTensorFlowConverter():
    unify_op_type_mapping = {
        "Conv2D": "conv2d",
        "DepthwiseConv2dNative": "conv2d",
        "MaxPool": "pooling",
        "AvgPool": "pooling",
        "ConcatV2": "concat",
        "MatMul": "matmul",
        "Pad": "pad"
    }

    def __init__(self):
        super().__init__()

        self.quantize_config = {'op_wise_config': {}}
        self.device = 'cpu'
        self.work_dir = os.path.abspath(logger.default_workspace)
        os.makedirs(self.work_dir, exist_ok=True)

        self.pre_optimized_model = None
        self.pre_optimizer_handle = None

        self.bf16_ops = []
        self.fp32_ops = []
        self.dump_times = 0   # for tensorboard

        cfg_yaml_name = "tensorflow.yaml"
        self.query_handler = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), cfg_yaml_name))
        self.op_wise_sequences = self.query_handler.get_eightbit_patterns()
        self.optimization = self.query_handler.get_grappler_optimization_cfg()

        self.fp32_results = []
        self.fp32_preds_as_label = False

    def _query_quantizable_ops(self, matched_nodes):
        """Collect the op-wise configuration for quantization.

        Returns:
            OrderDict: op-wise configuration.
        """
        uint8_type = self.query_handler.get_op_types_by_precision(precision='uint8')
        int8_type = self.query_handler.get_op_types_by_precision(precision='int8')
        tf_quantizable_op_type = list(set(uint8_type).union(set(int8_type)))

        valid_precision = self.query_handler.get_mixed_precision_combination()
        op_capability = self.query_handler.get_quantization_capability()
        conv_config = copy.deepcopy(op_capability['uint8']['Conv2D'])
        matmul_config = copy.deepcopy(op_capability['uint8']['MatMul'])
        other_config = copy.deepcopy(op_capability['uint8']['default'])
        if ('bf16' in valid_precision and CpuInfo().bf16) or os.getenv('FORCE_BF16') == '1':
            #TODO we need to enhance below logic by introducing precision priority.
            conv_config['weight']['dtype'].insert(-1, 'bf16')
            matmul_config['weight']['dtype'].insert(-1, 'bf16')
            conv_config['activation']['dtype'].insert(-1, 'bf16')
            matmul_config['activation']['dtype'].insert(-1, 'bf16')
            other_config['activation']['dtype'].insert(-1, 'bf16')

        self.quantizable_op_details = OrderedDict()

        self._init_op_stat = {i: [] for i in tf_quantizable_op_type}

        exclude_first_quantizable_op = False
        for details in matched_nodes:
            node_op = details[-1][0]
            node_name = details[0]
            patterns = details[-1]
            pat_length = len(patterns)
            pattern_info = {
                'sequence': [[','.join(patterns[:pat_length - i]) for i in range(pat_length)][0]],
                'precision': ['int8']
            }
            if node_op in tf_quantizable_op_type and node_name not in self.exclude_node_names and (
                node_name, self.unify_op_type_mapping[node_op]) not in self.quantizable_op_details:
                if exclude_first_quantizable_op and \
                    (self.unify_op_type_mapping[node_op].find("conv2d") != -1 or \
                    self.unify_op_type_mapping[node_op].find("matmul") != -1):
                    exclude_first_quantizable_op = False
                    self.exclude_node_names.append(node_name)
                    continue
                self._init_op_stat[node_op].append(node_name)
                if self.unify_op_type_mapping[node_op].find("conv2d") != -1:
                    conv2d_int8_config = copy.deepcopy(conv_config)
                    conv2d_int8_config['pattern'] = pattern_info
                    self.quantizable_op_details[(
                        node_name, self.unify_op_type_mapping[node_op]
                    )] = conv2d_int8_config
                elif self.unify_op_type_mapping[node_op].find("matmul") != -1:

                    matmul_int8_config = copy.deepcopy(matmul_config)
                    matmul_int8_config['pattern'] = pattern_info
                    # TODO enable the sym mode once the tf fixed the mkldequantize_op.cc bug.
                    # is_positive_input = self.pre_optimizer_handle.has_positive_input(node_name)
                    # matmul_scheme = 'sym' if is_positive_input else 'asym'
                    matmul_scheme = ['asym']
                    matmul_int8_config['activation']['scheme'] = matmul_scheme
                    self.quantizable_op_details[(
                        node_name, self.unify_op_type_mapping[node_op]
                    )] = matmul_int8_config
                else:
                    self.quantizable_op_details[(
                        node_name, self.unify_op_type_mapping[node_op]
                    )] = copy.deepcopy(other_config)

                self.quantize_config['op_wise_config'][node_name] = (False, "minmax", False)
        return self.quantizable_op_details

    def collect_biasadd_values(self, graph_def):
        import tensorflow as tf

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        sess = tf.compat.v1.Session(graph=graph)

        biasadds = {}
        with sess.as_default():
            for op in graph.get_operations():
                if op.type in ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']:
                    consumer = op.outputs[0].consumers()
                    if len(consumer) > 1:
                        continue

                    if consumer[0].type == 'Mul':
                        consumer = consumer[0].outputs[0].consumers()
                        if len(consumer) > 1:
                            continue

                    if consumer[0].type == 'BiasAdd':
                        biasadds[op.name] = consumer[0].inputs[1].eval()
        return biasadds

    def query_fw_capability(self, model):
        """Collect the model-wise and op-wise configuration for quantization.

        Args:
            model (tf.compat.v1.GraphDef): model definition.

        Returns:
            [dict]: model-wise & op-wise configuration for quantization.
        """
        from .graph_editor.graph_rewriter.generic.pre_optimize import PreOptimization

        self.pre_optimizer_handle = PreOptimization(model, self.optimization)

        self.pre_optimized_model, fold_batchnorm_scales = self.pre_optimizer_handle.get_optimized_model()

        bias_adds = self.collect_biasadd_values(self.pre_optimized_model.graph_def)

        model.graph_def = self.pre_optimized_model.graph_def

        self.exclude_node_names = self.pre_optimizer_handle.get_excluded_node_names()
        patterns = self.query_handler.generate_internal_patterns()
        matched_nodes = self.pre_optimizer_handle.get_matched_nodes(patterns)
        original_graph_node_name = [i.name for i in model.graph_def.node]
        matched_nodes = sorted(matched_nodes, reverse=True, key=lambda i: (
            original_graph_node_name.index(i[0]), len(i[-1])))

        def check_match(patterns, input_pattern):
            for i in patterns:
                if input_pattern == [i for i in i.replace('+', ' ').strip().split(' ') if i]:
                    return True
            return False

        copied_matched_nodes = copy.deepcopy(matched_nodes)
        for i in copied_matched_nodes:
            if i[-1][0] in self.query_handler.get_op_types()['int8']:
                continue

            if not self.pre_optimizer_handle.has_positive_input(i[0]) and \
                not check_match(self.query_handler.get_fuse_patterns()['int8'], i[-1]):
                print(f'Try to remove {i}')
                #matched_nodes.remove(i)

        del copied_matched_nodes

        self._query_quantizable_ops(matched_nodes)
        capability = {
            'optypewise': self.get_optype_wise_ability(),
        }
        capability['opwise'] = copy.deepcopy(self.quantizable_op_details)
        logger.debug("Dump framework quantization capability:")
        logger.debug(capability)

        return capability, fold_batchnorm_scales, bias_adds

    def get_optype_wise_ability(self):
        """Get the op type wise capability by generating the union value of each op type.
        Returns:
            [string dict]: the key is op type while the value is the
                           detail configurations of activation and weight for this op type.
        """
        res = OrderedDict()
        for op in self.quantizable_op_details:
            if op[1] not in res:
                res[op[1]] = {'activation': self.quantizable_op_details[op]['activation']}
                if 'weight' in self.quantizable_op_details[op]:
                    res[op[1]]['weight'] = self.quantizable_op_details[op]['weight']
        return res

    def analyze_qat_model(self, model):
        import tensorflow as tf

        input_signature = []
        for name, input in zip(model.input_names, model.inputs):
            input_signature.append(tf.TensorSpec(input.shape, input.dtype, name))
        concrete_function = tf.function(model).get_concrete_function(input_signature)
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        frozen_func = convert_variables_to_constants_v2(concrete_function, lower_control_flow=False)
        frozen_graph = frozen_func.graph

        sess = tf.compat.v1.Session(graph=frozen_graph)

        qat_model_parameters = {}
        qat_model_parameters['graph'] = frozen_graph
        qat_model_parameters['sess'] = sess

        with sess.as_default():
            fq_weights = {}
            conv_weights = {}
            mat_weights = {}
            for op in frozen_graph.get_operations():
                if op.type in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
                    consumer = op.outputs[0].consumers()
                    if len(consumer) > 1:
                        continue

                    if consumer[0].type in ['Reshape']:
                        consumer = consumer[0].outputs[0].consumers()
                        if len(consumer) > 1:
                            continue

                    if consumer[0].type in ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']:
                        fq_weights[consumer[0].name] = {
                            'min': op.inputs[1].eval(),
                            'max': op.inputs[2].eval(),
                        }
                if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                    conv_weights[op.name] = op.inputs[1].eval()

                if op.type == 'MatMul':
                    mat_weights[op.name] = op.inputs[1].eval()

            qat_model_parameters['fq_weights'] = fq_weights
            qat_model_parameters['conv_weights'] = conv_weights
            qat_model_parameters['mat_weights'] = mat_weights

        from tensorflow.python.training import saver
        from tensorflow.core.protobuf import config_pb2
        from tensorflow.python.grappler import tf_optimizer
        from tensorflow.core.protobuf import meta_graph_pb2
        graph_def = frozen_graph.as_graph_def()
        output_names = [output.split(':')[0] for output in model.output_names]
        # replace the output name with squential
        for output_name in output_names:
            for node in graph_def.node[::-1]:
                if node.op == 'Identity' and output_name in node.input[0]:
                    node.name = output_name
                    break

        grappler_meta_graph_def = saver.export_meta_graph(
            graph_def=graph_def, graph=frozen_graph)

        # Add a collection 'train_op' so that Grappler knows the outputs.
        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in model.output_names:
            fetch_collection.node_list.value.append(array)
        grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
            fetch_collection)
        grappler_session_config = config_pb2.ConfigProto()
        rewrite_options = grappler_session_config.graph_options.rewrite_options
        for item in ['pruning', 'shape', 'dependency', 'debug_stripper', 'loop', 'constfold', 'arithmetic']:
            rewrite_options.optimizers.append(item)
        rewrite_options.min_graph_nodes = -1
        const_fold_graph_def = tf_optimizer.OptimizeGraph(grappler_session_config,
                                                          grappler_meta_graph_def, graph_id=b"tf_graph")

        qat_model_parameters['const_fold_graph_def'] = const_fold_graph_def

        tf.io.write_graph(
            const_fold_graph_def,
            logger.default_workspace,
            'fp32_const_fold_graph_def.pb',
            as_text=False)

        return qat_model_parameters

    def convert(self, model):
        print(f'Working directory: {self.work_dir}')

        qat_model_parameters = self.analyze_qat_model(model._model)

        capability, fold_batchnorm_scales, bias_adds = self.query_fw_capability(model)

        qat_model_parameters['scales'] = fold_batchnorm_scales
        qat_model_parameters['bias_adds'] = bias_adds

        quantize_config = {'op_wise_config': {}}
        is_perchannel = False
        weight_bit = 7.0
        is_asymmetric = False
        algorithm = None
        for each_op_info in capability['opwise']:
            op_name = each_op_info[0]

            quantize_config['op_wise_config'][op_name] = (is_perchannel,
                                                          algorithm,
                                                          is_asymmetric,
                                                          weight_bit)
        from .graph_editor.graph_converter import GraphConverter
        converter = GraphConverter(model,
                                   qt_config=quantize_config,
                                   int8_sequences=self.op_wise_sequences,
                                   fake_quant=True,
                                   qat_model_parameters=qat_model_parameters)

        return converter.convert()


@singleton
class TensorflowQuery():

    def __init__(self, local_config_file=None):
        import tensorflow as tf

        super().__init__()
        self.version = tf.version.VERSION
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        default_config = None
        for sub_data in data:
            if self.version in sub_data['version']['name']:
                return sub_data

            if 'default' in sub_data['version']['name']:
                default_config = sub_data

        return default_config

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check if the format of {} follows LPOT yaml schema.".
                                 format(self.cfg))

    def get_version(self):
        """Get the current backend version infomation.

        Returns:
            [string]: version string.
        """
        return self.cur_config['version']['name']

    def get_precisions(self):
        """Get supported precisions for current backend.

        Returns:
            [string list]: the precisions' name.
        """
        return self.cur_config['precisions']['names']

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return self.cur_config['ops']

    def get_fuse_patterns(self):
        """Get supported patterns by low precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the supported patterns.
        """
        return self.cur_config['patterns']

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        return self.cur_config['capabilities']

    def get_op_types_by_precision(self, precision):
        """Get op types per precision

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        assert precision in list(self.cur_config['ops'].keys())

        return self.cur_config['ops'][precision]

    def get_mixed_precision_combination(self):
        """Get the valid mixed precisions.

        Returns:
            [string list]: valid precision list.
        """
        if self.cur_config['precisions']['valid_mixed_precisions']:
            return [i.strip() for i in self.cur_config['precisions']['valid_mixed_precisions']]

        return [i.strip() for i in self.get_precisions().split(',')]

    def get_grappler_optimization_cfg(self):
        return self.cur_config['grappler_optimization']

    def get_eightbit_patterns(self):
        """Get eightbit op wise sequences information.

        Returns:
            [dictionary]: key is the op type while value is the list of sequences start
                        with the op type same as key value.
        """
        quantizable_op_types = self.get_op_types_by_precision(
            'int8') + self.get_op_types_by_precision('uint8')
        int8_patterns = [i.replace(
            '+', ' ').split() for i in list(set(self.get_fuse_patterns()['int8'] +
                                                self.get_fuse_patterns()['uint8']))]

        res = {}
        for i in quantizable_op_types:
            res[i] = [[i]]

        for pattern in int8_patterns:
            op_type = pattern[0]
            if op_type in res:
                res[op_type].append(pattern)

        return res

    def generate_internal_patterns(self):
        """Translate the patterns defined in the yaml to internal pattern expression.
        """
        def _generate_pattern(data):
            length = [len(i) for i in data]
            res=[]
            for index in range(max(length)):
                if index <= min(length) - 1:
                    tmp = [i[index] for i in data]
                    if len(set(tmp)) == 1:
                        res.append([tmp[0]])
                    else:
                        res.append(tuple(set(tmp)))
                else:
                    tmp1 = [i[index] for i in data if len(i) > index]
                    res.append(tuple(set(tmp1)))

            return res

        op_level_sequences = {}

        for k, op_level_all_sequences in self.get_eightbit_patterns().items():
            op_level_sequences[k] = []
            sorted_sequences = sorted(op_level_all_sequences)
            last_len = 1
            each_combination = []
            for index, value in enumerate(sorted_sequences):
                if  len(value) >= last_len:
                    last_len = len(value)
                    each_combination.append(value)
                else:
                    op_level_sequences[k].append(copy.deepcopy(each_combination))
                    each_combination.clear()
                    each_combination.append(value)
                    last_len = len(value)

                if index == len(sorted_sequences) - 1:
                    op_level_sequences[k].append(copy.deepcopy(each_combination))

        final_out = []
        for _ , op_level_sequences in op_level_sequences.items():
            for similar_sequences in op_level_sequences:
                final_out.append(_generate_pattern(similar_sequences))

        return final_out
