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

import onnx

from nncf import Dataset
from nncf.common.quantization.structs import QuantizationPreset
from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.experimental.post_training.api.dataset import Dataset as PTQDataset



# It should be removed when we change all algorithms.
class CalibrationDataset(PTQDataset):
    """
    Wraps nncf.Dataset to make it suitable for ONNX post training experomantal API.
    """

    def __init__(self, dataset: Dataset):
        """
        :param dataloader: A `NNCFDataLoader` object.
        """
        super().__init__()
        self._dataset = dataset
        self._length = None
        self.

    def __len__(self) -> int:
        if self._length is None:
            self._length = CalibrationDataset._get_length(self._dataset.get_data())
        return self._length

    def __getitem__(self, index: int):
        item = self._dataloader[index]

        if isinstance(item, dict):
            for key in item:
                if not isinstance(item[key], np.ndarray):
                    raise RuntimeError('The input tensor should be numpy ndarray')
                item[key] = ONNXNNCFTensor(item[key])
        return item

    @staticmethod
    def _get_length(iterable) -> int:
        length = 0
        for _ in iterable:
            length = length + 1

        return length


def quantize_impl(model: onnx.ModelProto,
                  calibration_dataset: Dataset,
                  preset: str,
                  target_device: str,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[str] = None) -> onnx.ModelProto:
    """
    Implementation of the `quantize()` method for the ONNX backend.
    """
    if model_type is not None:
        raise ValueError(f'model_type={model_type} is not supported')
    if fast_bias_correction == False:
        raise ValueError(f'fast_bias_correction={fast_bias_correction} is not supported')

    builder = CompressionBuilder()

    quantization_parameters = PostTrainingQuantizationParameters(
        preset=QuantizationPreset.from_str(preset),
        target_device=target_device,
        number_samples=subset_size,
    )

    quantization = PostTrainingQuantization(quantization_parameters)
    builder.add_algorithm(quantization)

    onnx_dataset = ONNXDataset(calibration_dataset)
    quantized_model = builder.apply(model, onnx_dataset)

    return quantized_model
