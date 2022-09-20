"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Optional

import openvino
from openvino.tools import pot

from nncf.experimental.ptq.data.dataloader import NNCFDataLoader
from nncf.experimental.ptq.openvino.dataloader import POTDataLoader


def quantize_impl(model: openvino.runtime.Model,
                  calibration_dataset: NNCFDataLoader,
                  preset: str,
                  target_device: str,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[str] = None) -> openvino.runtime.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend.
    """
    ir_model_xml = '/tmp/model.xml'
    ir_model_bin = '/tmp/model.bin'
    openvino.offline_transformations.serialize(model, ir_model_xml, ir_model_bin)

    model_config = {
        'model_name': 'model',
        'model': ir_model_xml,
        'weights': ir_model_bin,
    }

    engine_config = {
        'device': 'CPU',
    }

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': target_device,
                'preset': preset,
                'stat_subset_size': subset_size,
                'use_fast_bias': fast_bias_correction,
                'model_type': model_type,
            }
        }
    ]

    pot_model = pot.load_model(model_config)
    pot_dataloader = POTDataLoader(calibration_dataset)
    engine = pot.IEEngine(engine_config, pot_dataloader)
    pipeline = pot.create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(pot_model)
    pot.compress_model_weights(compressed_model)

    compressed_model_paths = pot.save_model(compressed_model, save_path='/tmp/pot', model_name='model')
    ir_model_xml = compressed_model_paths[0]['model']
    ir_model_bin = compressed_model_paths[0]['weights']
    ie = openvino.runtime.Core()
    quantized_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    return quantized_model
