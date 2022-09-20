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

import numpy as np
import tensorflow as tf

from nncf import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.experimental.ptq.data.dataloader import NNCFDataLoader
from nncf.experimental.ptq.tensorflow.dataloader import PTQInitializingDataLoader
from nncf.tensorflow.helpers.model_creation import create_compressed_model


def quantize_impl(model: tf.Module,
                  calibration_dataset: NNCFDataLoader,
                  preset: str,
                  target_device: str,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[str] = None) -> tf.Module:
    """
    Implementation of the `quantize()` method for the PyTorch backend.
    """
    if model_type is not None:
        raise ValueError(f'model_type={model_type} is not supported')
    if fast_bias_correction == False:
        raise ValueError(f'fast_bias_correction={fast_bias_correction} is not supported')

    nncf_config = NNCFConfig(
        {
            "target_device": target_device,
            "compression": {
                "algorithm": "quantization",
                "preset": preset,
                "initializer": {
                    "range": {
                        "num_init_samples": subset_size
                    },
                    "batchnorm_adaptation": {
                        "num_bn_adaptation_samples": 0
                    }
                },
                "overflow_fix": "first_layer_only"
            }
        }
    )

    calibration_data_loader = PTQInitializingDataLoader(calibration_dataset)
    nncf_config.register_extra_structs(
        [
            QuantizationRangeInitArgs(data_loader=calibration_data_loader),
            BNAdaptationInitArgs(data_loader=calibration_data_loader)
        ]
    )

    compression_ctrl, compressed_model = create_compressed_model(
        model=model,
        config=nncf_config
    )
    stripped_model = compression_ctrl.strip_model(compressed_model)

    return stripped_model
