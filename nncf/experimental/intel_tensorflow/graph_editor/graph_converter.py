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

from collections import OrderedDict

import copy
import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from nncf.experimental.intel_tensorflow.model.model import Model
from .quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from .quantize_graph.quantize_graph_common import QuantizeGraphHelper

from .graph_rewriter.graph_util import GraphAnalyzer
from .graph_rewriter.generic.fuse_pad_with_conv import FusePadWithConv2DOptimizer

from .graph_rewriter.int8.fuse_conv_requantize import FuseConvRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeTransformer
from .graph_rewriter.int8.post_quantized_op_cse import PostCseOptimizer
from nncf.experimental.intel_tensorflow.graph_editor.graph_rewriter.int8.remove_fake_quant import RemoveFakeQuantOpOptimizer
from nncf.experimental.intel_tensorflow.graph_editor.quantize_graph.quantize_graph_common import QuantizeGraphHelper as helper

TF_SUPPORTED_MAX_VERSION = '2.6.0'
TF_SUPPORTED_MIN_VERSION = '1.14.0'

logger = logging.getLogger()
debug = bool(logger.level == logging.DEBUG)


class GraphConverter:
    def __init__(self,
                 model,
                 qt_config={},
                 recipes={},
                 int8_sequences={},
                 fp32_ops=[],
                 bf16_ops=[],
                 data_loader=None,
                 fake_quant=False,
                 itex_mode=False,
                 qat_model_parameters=None):
        """Convert graph.

        :param model: input tensorflow model.
        :param qt_config: quantization configs, including interation and op-wise quant config
        :param fp32_ops: fall back to fp32 dtype op list
        :param bf16_ops: fall back to bf16 dtype op list
        :param data_loader: for calibration phase used dataloader
        :param fake_quant: for quantization-aware training model conversion to default model
        """
        self.model = model
        #(TODO) does it right to make the internal model format as graph_def
        self.output_tensor_names = self.model.output_tensor_names
        self.input_tensor_names = self.model.input_tensor_names
        # quantize specific config
        self.calib_iteration = qt_config['calib_iteration'] if not fake_quant else 0
        self.op_wise_config = qt_config['op_wise_config']
        self.device = qt_config['device'] if 'device' in qt_config else 'cpu'
        self.int8_sequences = int8_sequences
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        self.recipes = recipes
        self.fake_quant = fake_quant
        self.itex_mode = itex_mode
        self.quantized_node_info = []
        self._calibration_data = []
        self._fp32_print_data = []
        self.data_loader = data_loader
        self._check_args()
        self._gen_tmp_filenames()
        self._kl_op_dict = {}
        self._kl_keys = []
        self._print_node_mapping = {}
        self._enable_kl_op_names = [
            k for k in self.op_wise_config if self.op_wise_config[k][1] == 'kl'
        ]
        self.scale_info = {}
        self.scale_info.update(qt_config)
        self.scale_info.update({'recipes': self.recipes})
        self.scale_info.update({'int8_sequences': self.int8_sequences})
        self.scale_info.update({'bf16_ops': self.bf16_ops})
        self.scale_info.update({'fp32_ops': self.fp32_ops})

        self._fp32_model = Model(self.model._model, **self.model.kwargs)
        self._fp32_model.graph_def = self.model.graph_def
        self._fp32_model.output_tensor_names = self.output_tensor_names
        self._fp32_model.input_tensor_names = self.input_tensor_names

        self._sampling_model = Model(self.model._model, **self.model.kwargs)
        self._sampling_model.output_tensor_names = self.output_tensor_names
        self._sampling_model.input_tensor_names = self.input_tensor_names

        self._itex_model = Model(self.model._model, **self.model.kwargs)
        self._itex_model.graph_def = self.model.graph_def
        self._itex_model.output_tensor_names = self.output_tensor_names
        self._itex_model.input_tensor_names = self.input_tensor_names
        self._tmp_graph_def = copy.deepcopy(self.model.graph_def)

        self._qat_model_parameters = qat_model_parameters

    def _check_args(self):
        if self.model.workspace_path and not os.path.isdir(self.model.workspace_path) \
                and not os.path.exists(os.path.dirname(self.model.workspace_path)):
            raise ValueError('"output_graph" directory does not exist.')
        self._output_path = self.model.workspace_path

    def _gen_tmp_filenames(self):
        self._int8_dynamic_range_model_path = os.path.join(self._output_path, \
                                                      'int8_dynamic_range_graph')
        self._int8_logged_model_path = os.path.join(self._output_path, 'int8_logged_graph')
        self._fp32_logged_model_path = os.path.join(self._output_path, 'fp32_logged_graph')
        self._int8_frozen_range_model_path = os.path.join(self._output_path,
                                                          'int8_frozen_range_graph')
        self._bf16_mixed_precision_model_path = os.path.join(self._output_path,
                                                        'int8_bf16_mixed_precision_graph')

        self.output_graph = os.path.join(self._output_path, 'int8_final_fused_graph')
        # to keep temp model
        self._tmp_model = Model(self.model._model, **self.model.kwargs)
        self._tmp_model.output_tensor_names = self.output_tensor_names
        self._tmp_model.input_tensor_names = self.input_tensor_names

    def convert(self):
        model = self.quantize()

        post_cse_graph_def = PostCseOptimizer(model.graph_def).do_transformation()
        post_cse_graph_def.library.CopyFrom(self.model.graph_def.library)
        model.graph_def = post_cse_graph_def

        if debug:
            model.save(self.output_graph)
            logger.info("Save converted graph file to {}.".format(self.output_graph))
        model.q_config = self.scale_info
        return model

    def remove_fake_quantize(self):
        self._tmp_graph_def = RemoveFakeQuantOpOptimizer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
        self._tmp_model.graph_def = self._tmp_graph_def

        # Debug
        from nncf.experimental.intel_tensorflow.utils.logger import default_workspace
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
            default_workspace,
            'fp32_remove_fake_model.pb',
            as_text=False)

    def _trace_graph(self, graph_def):
        graph_analyzer = GraphAnalyzer()
        graph_analyzer.graph = graph_def
        graph_info = graph_analyzer.parse_graph()

        trace = OrderedDict()

        stack = []
        for input_name in self.input_tensor_names:
            stack.append(input_name)

        visited = {}

        while stack:
            node_name = stack.pop()
            node_info = graph_info[node_name]

            if node_name in visited:
                visited[node_name] += 1
            else:
                visited[node_name] = 1 if node_info.node.input else 0
                for node_input in node_info.node.input:
                    node_input_name = node_input.split(':')[0]
                    if node_input_name in graph_info:
                        if graph_info[node_input_name].node.op == 'Const':
                            visited[node_name] += 1
                    else:
                        visited[node_name] += 1

            if visited[node_name] == len(node_info.node.input):
                trace[node_name] = node_info
                for output in node_info.outputs:
                    stack.append(output)

        return trace, graph_info

    def find_next_fq_parameters(self, graph_info, graph_trace):
        print('Find FQ parameters:')
        while True:
            node_name, node_info = graph_trace.popitem(last=False)
            print(f'FP32_GRAPH:{node_name} | {node_info.node.op}')
            if node_info.node.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
                narrow_range = node_info.node.attr['narrow_range'].b
                num_bits = node_info.node.attr['num_bits'].i
                min_node = graph_info[node_info.node.input[1]].node
                max_node = graph_info[node_info.node.input[2]].node
                min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
                max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
                q_type = tf.dtypes.qint8 if np.min(min_value) < 0 else tf.dtypes.quint8

                if q_type == tf.dtypes.qint8 and narrow_range == False:
                    print('Warning: type qint8, narrow_range = False')

                return min_value, max_value, q_type, num_bits

    def get_input_type(self, graph_info, node_name):
        if graph_info[node_name].node.op in ['Requantize']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['out_type'].type)
            min_node = graph_info[graph_info[node_name].node.input[3]].node
            max_node = graph_info[graph_info[node_name].node.input[4]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        if graph_info[node_name].node.op in ['QuantizedMatMulWithBiasAndReluAndRequantize']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['Toutput'].type)
            min_node = graph_info[graph_info[node_name].node.input[7]].node
            max_node = graph_info[graph_info[node_name].node.input[8]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        if graph_info[node_name].node.op in ['QuantizedConv2DWithBiasAndReluAndRequantize',
                                             'QuantizedConv2DWithBiasAndRequantize']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['out_type'].type)
            min_node = graph_info[graph_info[node_name].node.input[7]].node
            max_node = graph_info[graph_info[node_name].node.input[8]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        if graph_info[node_name].node.op in ['Quantize', 'QuantizeV2']:
            q_type = tf.dtypes.as_dtype(graph_info[node_name].node.attr['T'].type)
            min_node = graph_info[graph_info[node_name].node.input[1]].node
            max_node = graph_info[graph_info[node_name].node.input[2]].node
            min_value = tensor_util.MakeNdarray(min_node.attr['value'].tensor)
            max_value = tensor_util.MakeNdarray(max_node.attr['value'].tensor)
            return q_type, min_value, max_value

        return self.get_input_type(graph_info, graph_info[node_name].node.input[0])

    def _quantize(self, input, min_input, max_input, type):
        with tf.Graph().as_default() as quantized_graph:
            input_ = tf.compat.v1.placeholder(tf.float32, shape=input.shape, name='input')
            min_input_ = tf.constant(min_input, dtype=tf.float32, shape=min_input.shape)
            max_input_ = tf.constant(max_input, dtype=tf.float32, shape=max_input.shape)
            narrow_range = type == tf.dtypes.qint8
            axis = len(input.shape) - 1 if max_input.size > 1 else None
            q_input_, min_output_, max_output_ = tf.quantization.quantize(
                input_,
                min_input_,
                max_input_,
                type,
                mode='SCALED',
                round_mode='HALF_TO_EVEN',
                narrow_range=narrow_range,
                axis = axis,
                ensure_minimum_range=0.0)

            with tf.compat.v1.Session(graph=quantized_graph) as sess:
                out = sess.run(
                    [q_input_, min_output_, max_output_], feed_dict={input_: input})

                return tuple(out)

    def _generate_int32_bias_for_matmul(self, bias_tensor, input_range, filter_range):
        bias_scale = 255.0 * 127.0 / (input_range * filter_range)

        int32_bias =np.around(bias_tensor * bias_scale).astype('int32')

        return int32_bias

    def _fill_qat_parameters(self):
        quantized_graph_trace, quantized_graph_info = self._trace_graph(self._tmp_graph_def)
        fp32_graph_trace, fp32_graph_info = self._trace_graph(self._qat_model_parameters['const_fold_graph_def'])

        print('Fill QAT parameters:')
        for node_name, node_info in quantized_graph_trace.items():
            print(f'Q_GRAPH:{node_name} | {node_info.node.op}')
            if node_info.node.op in ['Quantize', 'QuantizeV2']:
                min, max, q_type, num_bits = self.find_next_fq_parameters(
                    fp32_graph_info,
                    fp32_graph_trace)

                axis = 3 if max.size > 1 else -1

                if num_bits != 8:
                    raise RuntimeError('Quantize: num_bits != 8')

                min_node = quantized_graph_info[node_info.node.input[1]].node
                max_node = quantized_graph_info[node_info.node.input[2]].node

                helper.set_attr_dtype(node_info.node, "T", q_type)
                helper.set_attr_string(node_info.node, "mode", b"SCALED")
                helper.set_attr_string(node_info.node, "round_mode", b"HALF_TO_EVEN")
                helper.set_attr_bool(node_info.node, "narrow_range", q_type == tf.dtypes.qint8)
                helper.set_attr_float(node_info.node, "ensure_minimum_range", 0.0)
                helper.set_attr_int(node_info.node, "axis", axis)

                helper.set_attr_dtype(min_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_node, "value", min, tf.dtypes.float32, min.shape)
                helper.set_attr_dtype(max_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_node, "value", max, tf.dtypes.float32, max.shape)


            if node_info.node.op in ['Requantize']:
                min, max, q_type, num_bits = self.find_next_fq_parameters(
                    fp32_graph_info,
                    fp32_graph_trace)

                if num_bits != 8:
                    raise RuntimeError('Requantize: num_bits != 8')

                min_node = quantized_graph_info[node_info.node.input[3]].node
                max_node = quantized_graph_info[node_info.node.input[4]].node

                helper.set_attr_dtype(node_info.node, "out_type", q_type)

                helper.set_attr_dtype(min_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_node, "value", min, tf.dtypes.float32, min.shape)
                helper.set_attr_dtype(max_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_node, "value", max, tf.dtypes.float32, max.shape)

            if node_info.node.op in [
                'QuantizedConv2DWithBiasSignedSumAndReluAndRequantize',
                'QuantizedConv2DWithBiasSumAndReluAndRequantize']:

                q_input_type, min_input, max_input = self.get_input_type(quantized_graph_info, node_info.node.input[0])
                helper.set_attr_dtype(node_info.node, "Tinput", q_input_type)
                helper.set_attr_dtype(node_info.node, "Tfilter", tf.dtypes.qint8)
                helper.set_attr_dtype(node_info.node, "Tbias", tf.dtypes.float32)

                q_summand_type, min_summand_input, max_summand_input = self.get_input_type(quantized_graph_info, node_info.node.input[9].split(':')[0])
                helper.set_attr_dtype(node_info.node, "Tsummand", q_summand_type)

                conv_name = node_info.node.name.replace('_eightbit_quantized_conv', '')
                conv_name = conv_name.replace('_eightbit_requantize', '')
                filter = self._qat_model_parameters['conv_weights'][conv_name]
                min_filter = self._qat_model_parameters['fq_weights'][conv_name]['min']
                max_filter = self._qat_model_parameters['fq_weights'][conv_name]['max']
                print(f'{conv_name} : min {min_filter}, max {max_filter}')
                q_filter, q_min, q_max = self._quantize(filter, min_filter, max_filter, tf.dtypes.qint8)
                bias = self._qat_model_parameters['bias_adds'][conv_name]
                if conv_name in self._qat_model_parameters['scales']:
                    scale = self._qat_model_parameters['scales'][conv_name]

                    # remove negative scales
                    negative_mul = np.ones(scale.shape, dtype=np.int8)
                    negative_mul[np.where(scale < 0)] = -1
                    q_filter = q_filter * negative_mul

                    scale[np.where(scale < 0)] *= -1.0

                    min_scaled_filter = min_filter * scale
                    max_scaled_filter = max_filter * scale
                else:
                    min_scaled_filter = min_filter
                    max_scaled_filter = max_filter

                filter_node = quantized_graph_info[node_info.node.input[1]].node
                bias_node = quantized_graph_info[node_info.node.input[2]].node
                min_filter_node = quantized_graph_info[node_info.node.input[5]].node
                max_filter_node = quantized_graph_info[node_info.node.input[6]].node

                helper.set_attr_dtype(filter_node, "dtype", tf.dtypes.qint8)
                helper.set_attr_tensor(filter_node, "value", q_filter, tf.dtypes.qint8, q_filter.shape)

                helper.set_attr_dtype(bias_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(bias_node, "value", bias, tf.dtypes.float32, bias.shape)

                helper.set_attr_dtype(min_filter_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_filter_node, "value", min_scaled_filter, tf.dtypes.float32,
                                       min_scaled_filter.shape)

                helper.set_attr_dtype(max_filter_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_filter_node, "value", max_scaled_filter, tf.dtypes.float32,
                                       max_scaled_filter.shape)

                re_min, re_max, req_type, num_bits = self.find_next_fq_parameters(
                    fp32_graph_info,
                    fp32_graph_trace)

                if node_info.node.op == 'QuantizedConv2DWithBiasSignedSumAndReluAndRequantize':
                    if num_bits != 7:
                        raise RuntimeError('QuantizedConv2DWithBiasSignedSumAndReluAndRequantize: num_bits != 7')
                    re_max = re_max * 255.0 / 127.0

                if node_info.node.op == 'QuantizedConv2DWithBiasSumAndReluAndRequantize':
                    if num_bits != 8:
                        raise RuntimeError('QuantizedConv2DWithBiasSignedSumAndReluAndRequantize: num_bits != 8')

                helper.set_attr_dtype(node_info.node, "out_type", req_type)

                min_freezed_output = quantized_graph_info[node_info.node.input[7]].node
                max_freezed_output = quantized_graph_info[node_info.node.input[8]].node

                helper.set_attr_dtype(min_freezed_output, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_freezed_output, "value", re_min, tf.dtypes.float32,
                                       re_min.shape)

                helper.set_attr_dtype(max_freezed_output, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_freezed_output, "value", re_max, tf.dtypes.float32,
                                       re_max.shape)

            if node_info.node.op in ['QuantizedConv2DWithBiasAndRelu',
                                     'QuantizedConv2DWithBiasAndReluAndRequantize',
                                     'QuantizedConv2DWithBias',
                                     'QuantizedConv2DWithBiasAndRequantize',
                                     'QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize']:
                q_input_type, min_input, max_input = self.get_input_type(quantized_graph_info, node_info.node.input[0])
                helper.set_attr_dtype(node_info.node, "Tinput", q_input_type)
                helper.set_attr_dtype(node_info.node, "Tfilter", tf.dtypes.qint8)
                helper.set_attr_dtype(node_info.node, "Tbias", tf.dtypes.float32)

                conv_name = node_info.node.name.replace('_eightbit_quantized_conv', '')
                conv_name = conv_name.replace('_eightbit_requantize', '')
                filter = self._qat_model_parameters['conv_weights'][conv_name]
                min_filter = self._qat_model_parameters['fq_weights'][conv_name]['min']
                max_filter = self._qat_model_parameters['fq_weights'][conv_name]['max']
                print(f'{conv_name} : min {min_filter}, max {max_filter}')
                if node_info.node.op in ['QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize']:
                    orig_shape = filter.shape
                    req_filter = np.reshape(filter, (orig_shape[0], orig_shape[1], orig_shape[2] * orig_shape[3]))
                    q_filter, q_min, q_max = self._quantize(req_filter, min_filter, max_filter, tf.dtypes.qint8)
                    q_filter = np.reshape(q_filter, orig_shape)
                else:
                    q_filter, q_min, q_max = self._quantize(filter, min_filter, max_filter, tf.dtypes.qint8)
                bias = self._qat_model_parameters['bias_adds'][conv_name]
                if conv_name in self._qat_model_parameters['scales']:
                    scale = self._qat_model_parameters['scales'][conv_name]

                    #remove negative scales
                    negative_mul = np.ones(scale.shape, dtype=np.int8)
                    negative_mul[np.where(scale < 0)] = -1
                    if node_info.node.op in ['QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize']:
                        orig_shape = q_filter.shape
                        req_filter = np.reshape(q_filter, (orig_shape[0], orig_shape[1], orig_shape[2] * orig_shape[3]))
                        req_filter = req_filter * negative_mul
                        q_filter = np.reshape(req_filter, orig_shape)
                    else:
                        q_filter = q_filter * negative_mul

                    scale[np.where(scale < 0)] *= -1.0

                    min_scaled_filter = min_filter * scale
                    max_scaled_filter = max_filter * scale
                else:
                    min_scaled_filter = min_filter
                    max_scaled_filter = max_filter

                filter_node = quantized_graph_info[node_info.node.input[1]].node
                bias_node = quantized_graph_info[node_info.node.input[2]].node
                min_filter_node = quantized_graph_info[node_info.node.input[5]].node
                max_filter_node = quantized_graph_info[node_info.node.input[6]].node

                helper.set_attr_dtype(filter_node, "dtype", tf.dtypes.qint8)
                helper.set_attr_tensor(filter_node, "value", q_filter, tf.dtypes.qint8, q_filter.shape)

                helper.set_attr_dtype(bias_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(bias_node, "value", bias, tf.dtypes.float32, bias.shape)

                helper.set_attr_dtype(min_filter_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_filter_node, "value", min_scaled_filter, tf.dtypes.float32, min_scaled_filter.shape)

                helper.set_attr_dtype(max_filter_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_filter_node, "value", max_scaled_filter, tf.dtypes.float32, max_scaled_filter.shape)

                if node_info.node.op in ['QuantizedConv2DWithBiasAndReluAndRequantize',
                                         'QuantizedConv2DWithBiasAndRequantize',
                                         'QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize']:
                    re_min, re_max, req_type, num_bits = self.find_next_fq_parameters(
                        fp32_graph_info,
                        fp32_graph_trace)
                    if num_bits != 8:
                        raise RuntimeError(f'{node_info.node.op}: num_bits != 8')

                    helper.set_attr_dtype(node_info.node, "out_type", req_type)

                    min_freezed_output = quantized_graph_info[node_info.node.input[7]].node
                    max_freezed_output = quantized_graph_info[node_info.node.input[8]].node

                    helper.set_attr_dtype(min_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(min_freezed_output, "value", re_min, tf.dtypes.float32,
                                           re_min.shape)

                    helper.set_attr_dtype(max_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(max_freezed_output, "value", re_max, tf.dtypes.float32,
                                           re_max.shape)

            if node_info.node.op in ['QuantizedMatMulWithBias',
                                     'QuantizedMatMulWithBiasAndReluAndRequantize']:
                q_input_type, min_input, max_input = self.get_input_type(quantized_graph_info, node_info.node.input[0])
                helper.set_attr_dtype(node_info.node, "T1", q_input_type)
                helper.set_attr_dtype(node_info.node, "T2", tf.dtypes.qint8)
                helper.set_attr_dtype(node_info.node, "Tbias", tf.dtypes.float32)
                helper.set_attr_string(node_info.node, "input_quant_mode", b"SCALED")

                matmul_name = node_info.node.name.replace('_eightbit_quantized_mat_mul', '')
                matmul_name = matmul_name.replace('_eightbit_requantize', '')
                b = self._qat_model_parameters['mat_weights'][matmul_name]
                min_filter = self._qat_model_parameters['fq_weights'][matmul_name]['min']
                max_filter = self._qat_model_parameters['fq_weights'][matmul_name]['max']
                print(f'{matmul_name} : min {min_filter}, max {max_filter}')
                q_b, q_min, q_max = self._quantize(b, min_filter, max_filter, tf.dtypes.qint8)

                bias = self._qat_model_parameters['bias_adds'][matmul_name]

                b_node = quantized_graph_info[node_info.node.input[1]].node
                bias_node = quantized_graph_info[node_info.node.input[2]].node
                min_b_node = quantized_graph_info[node_info.node.input[5]].node
                max_b_node = quantized_graph_info[node_info.node.input[6]].node

                helper.set_attr_dtype(bias_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(bias_node, "value", bias, tf.dtypes.float32, bias.shape)

                helper.set_attr_dtype(b_node, "dtype", tf.dtypes.qint8)
                helper.set_attr_tensor(b_node, "value", q_b, tf.dtypes.qint8, q_b.shape)

                helper.set_attr_dtype(min_b_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(min_b_node, "value", min_filter, tf.dtypes.float32,
                                       min_filter.shape)

                helper.set_attr_dtype(max_b_node, "dtype", tf.dtypes.float32)
                helper.set_attr_tensor(max_b_node, "value", max_filter, tf.dtypes.float32,
                                       max_filter.shape)

                if node_info.node.op in ['QuantizedMatMulWithBiasAndReluAndRequantize']:
                    re_min, re_max, req_type, num_bits = self.find_next_fq_parameters(
                        fp32_graph_info,
                        fp32_graph_trace)
                    if num_bits != 8:
                        raise RuntimeError(f'{node_info.node.op}: num_bits != 8')

                    helper.set_attr_dtype(node_info.node, "Toutput", req_type)

                    min_freezed_output = quantized_graph_info[node_info.node.input[7]].node
                    max_freezed_output = quantized_graph_info[node_info.node.input[8]].node

                    helper.set_attr_dtype(min_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(min_freezed_output, "value", re_min, tf.dtypes.float32,
                                           re_min.shape)

                    helper.set_attr_dtype(max_freezed_output, "dtype", tf.dtypes.float32)
                    helper.set_attr_tensor(max_freezed_output, "value", re_max, tf.dtypes.float32,
                                           re_max.shape)

                    if q_input_type != tf.quint8:
                        raise RuntimeError(f'Input type is tf.qint8 for {node_info.node.name}. tf.quint8 is expected')
                    filter_range = np.maximum(np.abs(min_filter), np.abs(max_filter))
                    input_range = max_input
                    bias_int32 = self._generate_int32_bias_for_matmul(bias, input_range, filter_range)

                    helper.set_attr_dtype(node_info.node, "Tbias", tf.dtypes.qint32)
                    helper.set_attr_dtype(bias_node, "dtype", tf.dtypes.qint32)
                    helper.set_attr_tensor(bias_node, "value", bias_int32, tf.dtypes.qint32, bias_int32.shape)

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

        # Debug
        from nncf.experimental.intel_tensorflow.utils.logger import default_workspace
        tf.io.write_graph(
            self._tmp_graph_def,
            default_workspace,
            'fill_qat_parameters.pb',
            as_text=False)

    def _fuse_requantize(self):
        self._tmp_graph_def = FuseConvRequantizeTransformer(
            self._tmp_graph_def,
            self.device).do_transformation()

        self._tmp_graph_def = FuseMatMulRequantizeTransformer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

        # Debug
        from nncf.experimental.intel_tensorflow.utils.logger import default_workspace
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
            default_workspace,
            'fuse_requantize.pb',
            as_text=False)

    def quantize(self):
        """Quantize graph only (without optimizing fp32 graph), including:
            1) quantize graph,
            2) calibration,
            3) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        if not self.fake_quant:
            raise RuntimeError('fake_quant flag must be True')

        try:
            self.remove_fake_quantize()

            self._quantize_graph()

            self._fuse_requantize()
            self._fill_qat_parameters()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._tmp_model = None
            logger.error("Fail to quantize graph due to {}.".format(str(e)))
        finally:
            if not debug:
                self._post_clean()
            return self._tmp_model

    def _quantize_graph(self):
        """quantize graph."""

        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))

        #Debug
        from nncf.experimental.intel_tensorflow.utils.logger import default_workspace
        import tensorflow as tf
        tf.io.write_graph(
            self._tmp_graph_def,
            default_workspace,
            'fp32_pre_quantized_model.pb',
            as_text=False)

        self._tmp_graph_def = FusePadWithConv2DOptimizer(
            self._tmp_graph_def,
            non_pad_ops,
            self._tmp_model.input_node_names,
            self.op_wise_config).do_transformation()

        self._tmp_graph_def = QuantizeGraphHelper().get_sorted_graph(
            self._tmp_graph_def,
            self._tmp_model.input_node_names,
            self._tmp_model.output_node_names)

        self._tmp_graph_def, self.quantized_node_info = QuantizeGraphForIntel(
            self._tmp_graph_def,
            self._tmp_model.output_node_names,
            self.op_wise_config,
            self.int8_sequences,
            self.device,
            self.fake_quant).do_transform()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
        if debug:
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_dynamic_range_model_path)

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None
        """
        if os.path.exists(self._int8_logged_model_path) and \
            os.path.isdir(self._int8_logged_model_path):
            import shutil
            shutil.rmtree(self._int8_logged_model_path)

        elif gfile.Exists(self._int8_logged_model_path + '.pb'):
            os.remove(self._int8_logged_model_path + '.pb')
