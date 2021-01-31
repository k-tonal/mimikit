from copy import copy

from ..data.data_object import DataObject


class DSWrapper(DataObject):

    def __init__(self):
        pass

    def upgrade(self, dataset):
        """
        dynamically extends an object with own methods and attributes.
        @param dataset: object to be extended
        @return: extended object will be an instance of object.__class__ and of self.__class__
        """
        new = copy(dataset)
        bases = (self.__class__, new.__class__)
        name = self.__class__.__name__ + "Dataset"
        new.__dict__.update(self.__dict__)
        new.__class__ = type(name, bases, new.__dict__)
        return new

    def __call__(self, dataset: DataObject):
        return self.upgrade(dataset)


class InputEqualTarget(DSWrapper):
    def __getitem__(self, item):
        return tuple(self.data[i] for i in [item, item])

    def __len__(self):
        return len(self.data)


class ShiftedSeqsPair(DSWrapper):
    def __init__(self, input_length, targets, stride=1):
        """
        @param input_length:
        @param targets: [(shift, length)... ]
        @param stride:
        """
        self.shifts, self.lengths = list(zip(*[(0, input_length), *targets]))
        self.stride = stride
        self.N = None

    def __call__(self, dataset: DataObject):
        # grab the number of time-steps BEFORE we upgrade
        self.N = len(dataset.data)
        return super(ShiftedSeqsPair, self).__call__(dataset)

    def __len__(self):
        ln = (self.N - max(shift + ln for shift, ln in zip(self.shifts, self.lengths)) + 1) // self.stride
        return ln

    def __getitem__(self, item):
        i = item * self.stride
        items_slices = [slice(i + shift, i + shift + length)
                         for shift, length in zip(self.shifts, self.lengths)]
        return tuple(self.data[idx] for idx in items_slices)


