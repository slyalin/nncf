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

from typing import Optional
from nncf.experimental.intel_tensorflow.model.model import Model
from nncf.experimental.intel_tensorflow.converter import IntelTensorFlowConverter

from nncf.common.exporter import Exporter


class IntelTensorFlowExporter(Exporter):
    """
    This class provides export of the compressed model to the Intel TensorFlow
    SavedModel.
    """

    _SAVED_MODEL_FORMAT = 'tf'

    def export_model(self, save_path: str, save_format: Optional[str] = None) -> None:
        """
        Exports the compressed model to the specified format.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format. The default format is `tf` for export
            to the Intel Tensorflow SavedModel format.
        """
        if save_format is None:
            save_format = IntelTensorFlowExporter._SAVED_MODEL_FORMAT

        if save_format != IntelTensorFlowExporter._SAVED_MODEL_FORMAT:
            raise ValueError('IntelTFExporter suports only the Saved Model format')

        model_converter = IntelTensorFlowConverter()
        model = Model(self._model)
        converted_model = model_converter.convert(model)
        converted_model.save(save_path)
