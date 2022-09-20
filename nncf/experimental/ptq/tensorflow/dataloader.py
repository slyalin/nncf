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

from typing import Any, Tuple, Dict

import tensorflow as tf

from nncf.experimental.ptq.data.dataloader import NNCFDataLoader as PTQNNCFDataLoader
from nncf.common.initialization.dataloader import NNCFDataLoader


class PTQInitializingDataLoader(NNCFDataLoader):
    """
    This class wraps the tf.data.Dataset class.

    This is required for proper initialization of certain compression algorithms.
    """

    def __init__(self, data_loader: PTQNNCFDataLoader):
        self._data_loader = data_loader

        self._num  = 0
        self._max = len(self._data_loader)

    @property
    def batch_size(self) -> int:
        if not hasattr(self._data_loader, '_data_source'):
            return 1
        data_source = getattr(self._data_loader, '_data_source')

        if not hasattr(data_source, '_batch_size'):
            return 1
        batch_size = getattr(data_source, '_batch_size')
        try:
            if isinstance(batch_size, tf.Tensor):
                batch_size = batch_size.numpy()
            batch_size = int(batch_size)
        except:
            batch_size = 1
        return batch_size

    def __getitem__(self, index: int):
        data_item = self._data_loader[index]
        return (data_item, None)

    def __iter__(self):
        self._num = 0
        return self

    def __next__(self):
        if (self._num >= self._max):
            raise StopIteration

        output = self[self._num]
        self._num += 1

        return output
