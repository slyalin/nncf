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
from typing import Callable
from typing import TypeVar
from abc import ABC
from abc import abstractmethod


DataSource = TypeVar('DataSource')


class NNCFDataLoader(ABC):
    """
    The `NNCFDataLoader` provides input samples for the model.
    The class describes the interface that is used by compression algorithms.
    """

    @abstractmethod
    def __len__(self):
        """
        Returns the number of elements in the dataset.

        :return: A number of elements in the dataset.
        """

    @abstractmethod
    def __getitem__(self, index: int):
        """
        Returns the `index`-th data item from the dataset.

        :param index: An integer index in the range from `0` to `len(self)`.
        :return: The `index`-th data item from the dataset.
        """


def create_dataloader(data_source: DataSource,
                      transform_fn: Optional[Callable] = None) -> NNCFDataLoader:
    """
    Wraps a provided custom data source into `NNCFDataLoader` and makes
    it suitable for use in compression algorithms.

    :param data_source: An iterable python object. For more details
        please see [iterable](https://docs.python.org/3/glossary.html#term-iterable).
    :param transform_fn: A function that takes a data item from custom data source
        and makes it suitable for model inference.
    :return: A `NNCFDataLoader` object which wraps the custom data source.
    """
    if hasattr(data_source, '__getitem__'):
        return NNCFDataLoaderMapStyleImpl(data_source, transform_fn)

    if hasattr(data_source, '__iter__'):
        return NNCFDataLoaderIterableImpl(data_source, transform_fn)

    raise ValueError('Unsupported type of data source.')


class NNCFDataLoaderMapStyleImpl(NNCFDataLoader):
    """
    Wraps map-style data source into `NNCFDataLoader` object and makes
    data items from it suitable for model inference. A map-style data
    source is one that implements the `__getitem__()` and `__len__()`
    protocols and represents a map from integer indices to data samples.
    """

    def __init__(self,
                 data_source: DataSource,
                 transform_fn: Optional[Callable] = None):
        """
        Initializes a wrapper for a map-style data source.

        :param data_source: A map-style data source.
        :param transform_fn: A method that takes a data item from the data
            source and transforms it into the model expected input.
        """
        self._data_source = data_source
        self._transform_fn = transform_fn

    def __getitem__(self, index: int):
        sample = self._data_source[index]
        if self._transform_fn:
            sample = self._transform_fn(sample)
        return sample

    def __len__(self):
        return len(self._data_source)


# TODO(andrey-churkin): Implementation should be tested.
# We should check that everything works for the TensorFlow backend.
class NNCFDataLoaderIterableImpl(NNCFDataLoader):
    """
    Wraps iterable-style data source into `NNCFDataLoader` object and
    makes data items from it suitable for model inference. An iterable-style
    data source is one that implements `__iter__()` and represents an iterable
    object over data samples.
    """

    def __init__(self,
                 data_source: DataSource,
                 transform_fn: Optional[Callable] = None):
        """
        Initializes a wrapper for a iterable-style data source.

        :param data_source: An iterable-style data source.
        :param transform_fn: A method that takes a data item from the data
            source and transforms it into the model expected input.
        """
        self._data_source = data_source
        self._transform_fn = transform_fn

        self._size = None
        self._it = None
        self._elem = None
        self._elem_idx = -1

    def __getitem__(self, index: int):
        if index == self._elem_idx:
            return self._elem

        if self._it is None or index < self._elem_idx:
            self._it = iter(self._data_source)
            self._elem_idx = -1

        while self._elem_idx != index:
            try:
                self._elem = next(self._it)
                self._elem_idx = self._elem_idx + 1
            except StopIteration:
                self._it = None
                self._elem = None
                self._elem_idx = -1
                break

        if self._elem is None:
            raise IndexError('Index out of range.')

        output = self._elem
        if self._transform_fn is not None:
            output = self._transform_fn(output)

        return output

    def __len__(self):
        if self._size is None:
            self._size = len(self._data_source)
        return self._size
