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

from openvino.tools import pot

from nncf.experimental.ptq.data.dataloader import NNCFDataLoader


class POTDataLoader(pot.DataLoader):
    """
    Wraps NNCFDataloader to make it suitable for post-training optimization tool.
    """

    def __init__(self, dataloader: NNCFDataLoader):
        """
        Initializes a `POTDataLoader` instance.

        :param dataloader: A `NNCFDataLoader` object.
        """
        super().__init__(config=None)
        self._dataloader = dataloader

    def __len__(self):
        return len(self._dataloader)

    def __getitem__(self, index: int):
        return (self._dataloader[index], None)
