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

import numpy as np

from nncf.experimental.onnx.tensor import ONNXNNCFTensor
from nncf.experimental.ptq.data.dataloader import NNCFDataLoader
from nncf.experimental.post_training.api.dataset import Dataset


class ONNXDataset(Dataset):
    """
    Wraps NNCFDataloader to make it suitable for ONNX post training experomantal API.
    """

    def __init__(self, dataloader: NNCFDataLoader):
        """
        Initializes a `ONNXDataset` instance.

        :param dataloader: A `NNCFDataLoader` object.
        """
        super().__init__()
        self._dataloader = dataloader

    def __len__(self):
        return len(self._dataloader)

    def __getitem__(self, index: int):
        item = self._dataloader[index]

        if isinstance(item, dict):
            for key in item:
                if not isinstance(item[key], np.ndarray):
                    raise RuntimeError('The input tensor should be numpy ndarray')
                item[key] = ONNXNNCFTensor(item[key])
        return item
