import torch
from torch.utils.data import Sampler, \
    Dataset, IterableDataset, TensorDataset, Subset, DataLoader, \
    random_split
import pytorch_lightning as pl
import numpy as np
from typing import Sequence, Tuple, Generator, NewType, Union
from multiprocessing import cpu_count
from . import FeatureProxy
from .metadata import Metadata


# Dataset (map-style) is a parent of IterableDataset, so we need a strong test
def is_map_style_dataset(dataset):
    tp = type(dataset)
    return issubclass(tp, Dataset) and not issubclass(tp, IterableDataset)


def is_wrapper_ds(dataset):
    return issubclass(type(dataset), DatasetWrapper) or issubclass(type(dataset), IterableDatasetWrapper)


def zip_stack(batch_lists):
    """
    returns (stack(feat_1), stack(feat_2)...)
    """
    rv = tuple(torch.stack(x) for x in zip(*batch_lists))
    return rv


def maybe_not_tensor(x):
    x_class = type(x)
    if issubclass(x_class, torch.Tensor):
        pass
    elif issubclass(x_class, np.ndarray):
        x = torch.from_numpy(x)
    else:
        # First, try to instantiate a tensor from feat, then try to select the subset
        try:
            x = torch.tensor(x, requires_grad=False)
        except Exception as e:
            raise ValueError("Couldn't instantiate a `torch.Tensor` with the provided `feature` argument."
                             "Call to torch.tensor raised following Exception : \n" + str(e))
    return x


class SingleTensorDataset(TensorDataset):
    """
    torch's class has a getitem that returns tuples of tuples...
    """
    def __getitem__(self, item):
        rv = super(SingleTensorDataset, self).__getitem__(item)
        return rv[0]


class H5Dataset(Dataset):
    """
    just wraps a FeatureProxy in a Dataset class
    """

    def __init__(self, feature_proxy):
        self.ds = feature_proxy

    def __getitem__(self, item):
        return torch.from_numpy(self.ds[item])

    def __len__(self):
        return len(self.ds)


class GeneratorDataset(IterableDataset):
    """
    just wrap a Generator in an IterableDataset
    """

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)


class DatasetWrapper(Dataset):
    """
    Base wrapper for map-style Datasets.
    the gist of this kind of wrapper is to be able to configure the wrapping in __init__
    and then to bind the datasets int the wrapper through the __call__ method.
    E.g. :
    features = (Dataset(inputs), Dataset(targets))
    wrapper = MaybeScrambledTargetsWrapper(pr_scramble=0.5)
    dataset = wrapper(*features)

    This base class could be named 'ZipDataset' (in the spirit of torch's 'ConcatDataset') since
    __getitem__ packs data-points from several datasets, but for the same item, in a tuple (ds_1[i], ds_2[i], ...)

    IMPORTANT NOTE: Please make sure to call the super's __call__ method when subclassing
    in order to validate the datasets passed as arguments or you'll likely end up with exceptions all over the place!
    """

    def __init__(self, *args, **kwargs):
        self.datasets = None

    def __call__(self, *datasets: Dataset):
        if not all(is_map_style_dataset(ds) for ds in datasets):
            raise ValueError("Expected all datasets to be map-style datasets, got following types: %s"
                             % str([type(ds) for ds in datasets]))
        assert all(len(datasets[0]) == len(ds) for ds in datasets), "All Datasets must have the same length."
        self.datasets = datasets
        return self

    def __getitem__(self, item):
        return tuple(ds[item] for ds in self.datasets)

    def __len__(self):
        return len(self.datasets[0])


class IterableDatasetWrapper(IterableDataset):
    """
    Base wrapper for IterableDatasets.

    IMPORTANT NOTE: Please make sure to call the super's __call__ method when subclassing
    in order to validate the datasets passed as arguments or you'll likely end up with exceptions all over the place!
    """

    def __init__(self, *args, **kwargs):
        self.datasets = None

    def __call__(self, *datasets: IterableDataset):
        if not all(issubclass(type(ds), IterableDataset) for ds in datasets):
            raise ValueError("Expected all datasets to be subclasses of `IterableDataset`, got following types: %s"
                             % str([type(ds) for ds in datasets]))
        self.datasets = datasets
        return self

    def __iter__(self):
        return iter(zip(*tuple(iter(ds) for ds in self.datasets)))


class AutoEncodingWrapper(DatasetWrapper):
    """
    Duplicates the pointer to a Dataset, to serve batches where inputs == targets efficiently (in terms of memory)
    """
    def __call__(self, *datasets: Dataset):
        super(AutoEncodingWrapper, self).__call__(*datasets)
        if len(self.datasets) > 1:
            raise ValueError("Expected only one Dataset, got %i." % len(datasets))
        self.datasets = tuple([self.datasets[0]] * 2)
        return self


class SequenceModelWrapper(AutoEncodingWrapper):
    """
    First wraps a Feature in an AutoEncodingWrapper, then maps `item` in __getitem__ to shifted slices.
    for a standard language-modelling task, one would just need to do :
    wrapper = SequenceModelWrapper(sequence_length=128, shift=1)
    dataset = wrapper(Dataset(my_texts))
    """
    def __init__(self, sequence_length, shift, stride=1):
        super(SequenceModelWrapper, self).__init__()
        self.shift = shift
        self.sequence_length = sequence_length
        self.stride = stride

    def _item_to_slices(self, item):
        # TODO
        pass

    def __getitem__(self, item):
        return tuple(ds[i] for i, ds in zip([item, item+self.shift], self.datasets))


Feature = NewType("Feature",
                  Union[Dataset, IterableDataset, FeatureProxy,
                        np.ndarray, torch.Tensor, Sequence, Generator])


class DataModuleBase(pl.LightningDataModule):

    def __init__(self,
                 feature: [Feature, Tuple[Feature]],
                 subset_idx: [np.ndarray, torch.Tensor, Metadata, Sequence] = None,
                 malloc_device: [torch.device, str, Sequence] = None,
                 ds_wrapper: [DatasetWrapper, IterableDatasetWrapper] = None,
                 train_val_split=1.,
                 **loader_kwargs,
                 ):
        super(DataModuleBase, self).__init__()
        self.feature = feature
        self.subset_idx = subset_idx
        self.malloc_device = malloc_device
        self.ds_wrapper = ds_wrapper
        self.train_val_split = train_val_split
        self.loader_kwargs = loader_kwargs
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None

    def prepare_data(self, *args, **kwargs):
        # Reminder : do not assign state in this method (e.g. self.x = y)
        pass

    def setup(self, stage=None):
        dataset = self._prepare_feature()
        if is_map_style_dataset(dataset):
            tr_length = int(self.train_val_split * len(dataset))
            lengths = (tr_length, len(dataset) - tr_length)
            self.train_ds, self.val_ds = random_split(dataset, lengths)
        else:  # it's an iterable, we can't split...
            self.train_ds, self.val_ds = dataset, None
        self.loader_kwargs = self._set_defaults_loader_kwargs(dataset)

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train_ds, **self.loader_kwargs)

    def val_dataloader(self, *args, **kwargs):
        if self.val_ds is not None:
            return DataLoader(self.val_ds, **self.loader_kwargs)
        return None

    def test_dataloader(self, *args, **kwargs):
        if self.test_ds is not None:
            return DataLoader(self.test_ds, **self.loader_kwargs)
        return None

    def transfer_batch_to_device(self, batch, device):
        if isinstance(batch, tuple):
            batch = tuple(tensor.to(device) for tensor in batch)
        else:
            raise TypeError("Expected `batch` to be a tuple of tensors got type : '%s'" % str(type(batch)))
        return batch

    ###############################
    # Machinery to prepare features :
    ###############################

    def _prepare_feature(self):
        """
        this is the high-level method (called in setup()) that "converts" whatever the user passed as `feature` argument
        into a valid Dataset for torch's Dataloader.
        if `feature` is a tuple of Feature, it will first convert each element into a Dataset and the resulting Datasets
        will be wrapped into a DatasetWrapper.
        For details about the conversion process, please refer to the next method `_prepare_single_feature`

        @return: `self.feature` packed into a subclass of Dataset
        """
        feature, subset_idx = self.feature, self.subset_idx
        malloc_device, ds_wrapper = self.malloc_device, self.ds_wrapper

        if isinstance(feature, tuple):
            dataset = tuple(self._prepare_single_feature(feat, subset_idx, malloc_device) for feat in feature)
        else:
            dataset = self._prepare_single_feature(feature, subset_idx, malloc_device)
    
        if ds_wrapper is not None:
            if not isinstance(dataset, tuple):
                dataset = (dataset,)
            if self._is_valid_ds_tuple(dataset):
                dataset = ds_wrapper(*dataset)
        elif ds_wrapper is None and isinstance(dataset, tuple) and self._is_valid_ds_tuple(dataset):
            # by default, pack multiple features in a default wrapper
            if is_map_style_dataset(dataset[0]):
                dataset = DatasetWrapper()(*dataset)
            else:
                dataset = IterableDatasetWrapper()(*dataset)

        return dataset

    def _prepare_single_feature(self, feature, subset_idx, malloc_device):
        """
        in pseudo-code:
        if malloc:
            feature = tensor(feature)[subset].to(device)
        dataset = Dataset(feature)
        if subset and not malloc:
            dataset = Subset(dataset, subset)
        return dataset
        """
        if malloc_device is not None:
            if issubclass(type(feature), Dataset):
                raise ValueError("Cannot allocate a %s to a device's memory." % type(feature) +
                                 " Please set `malloc_device` to None " 
                                 "or pass a `feature` object of one of the following type: "
                                 "mmk.data.FeatureProxy, np.ndarray, torch.Tensor, Sequence")
            feature = self._feature_to_tensor(feature, subset_idx)
            feature = self._alloc_feature(feature, malloc_device)

        dataset = self._feature_to_dataset(feature)

        if malloc_device is None and subset_idx is not None:
            if issubclass(type(dataset), IterableDataset):
                raise ValueError("Cannot take a subset of an IterableDataset. Please set `subset_idx` to None or "
                                 "pass a feature of another class.")
            subset_idx = DataModuleBase._subset_idx_to_indexing_object(subset_idx)
            dataset = Subset(dataset, subset_idx)

        return dataset

    @staticmethod
    def _feature_to_tensor(feature, subset_idx):
        feat_class = type(feature)
        subset_idx = DataModuleBase._subset_idx_to_indexing_object(subset_idx)

        # because h5 feature can be huge and need sorted indices,
        # we single this case out and return the "subseted" feature right away
        if issubclass(feat_class, FeatureProxy):
            if isinstance(subset_idx, Sequence):
                subset_idx = np.sort(subset_idx)
            return torch.from_numpy(feature[subset_idx])

        # otherwise, we first attempt to build a tensor before taking the subset
        feature = maybe_not_tensor(feature)

        return feature[subset_idx]

    @staticmethod
    def _alloc_feature(feature, device):
        if device is None:
            return
        if not isinstance(device, torch.device):
            if not isinstance(device, str) and isinstance(device, Sequence):
                device = torch.device(*device)
            else:
                try:
                    device = torch.device(device)
                except Exception as e:
                    raise ValueError("Couldn't instantiate a `torch.device` with the provided `malloc_device` argument."
                                     "Call to torch.device raised following Exception : \n" + str(e))
        return feature.to(device)

    @staticmethod
    def _feature_to_dataset(feature):
        feat_class = type(feature)
        if issubclass(feat_class, Dataset):
            dataset = feature
        elif issubclass(feat_class, torch.Tensor):
            dataset = SingleTensorDataset(feature)
        elif issubclass(feat_class, np.ndarray):
            dataset = SingleTensorDataset(torch.from_numpy(feature))
        elif issubclass(feat_class, FeatureProxy):
            dataset = H5Dataset(feature)
        elif issubclass(feat_class, Sequence):
            dataset = SingleTensorDataset(torch.tensor(feature, requires_grad=False))
        elif issubclass(feat_class, Generator):
            dataset = GeneratorDataset(feature)
        else:
            raise TypeError("Cannot instantiate a Dataset with a `feature` object"
                            " of type `" + str(feat_class) + "`")
        return dataset

    @staticmethod
    def _wrap_dataset(feat, wrapper):
        if issubclass(wrapper, DatasetWrapper):
            if not isinstance(feat, tuple):
                feat = (feat, )
            feat = wrapper(*feat)
        else:
            raise TypeError("Expected `ds_wrapper` to be a subclass of `DatasetWrapper` or `IterableDatasetWrapper`,"
                            " got : `%s`" % str(type(wrapper)))
        return feat

    @staticmethod
    def _subset_idx_to_indexing_object(subset_idx):
        if subset_idx is None:
            subset_idx = slice(None, None, None)
        elif isinstance(subset_idx, Metadata):
            subset_idx = subset_idx.all_indices
        # hopefully, those cover all valid cases...
        elif type(subset_idx) in (np.ndarray, list, torch.Tensor, slice, tuple):
            pass
        else:
            raise TypeError("Cannot construct a valid indexing object"
                            " with the provided `subset_idx` argument of type `" +
                            str(type(subset_idx)) + "`")
        return subset_idx

    @staticmethod
    def _is_valid_ds_tuple(datasets):
        types = [type(ds) for ds in datasets]
        if any(is_map_style_dataset(ds) for ds in datasets) \
                and any(issubclass(t, IterableDataset) for t in types):
            raise ValueError("argument `feature` cannot mix map-style and iterable datasets.")
        return True

    def _set_defaults_loader_kwargs(self, dataset):
        """
        if the user didn't explicitly set a parameter that needs to (or should) be set, we do it for him.
        Specifically:
            - if the ds will return Sequences of Features, we need an appropriate collate_fn
            - if the ds isn't mallocated, we should use several workers and pin_memory
        """
        kwargs = self.loader_kwargs
        # if the batch_size is explicitly None, we assume that the user doesn't want us to fiddle with his batch
        # regardless whether or not he specified a collate_fn.
        if is_wrapper_ds(dataset) and kwargs.get("batch_size", 1) is not None:
            kwargs.setdefault("collate_fn", zip_stack)
        if self.malloc_device is None:
            kwargs.setdefault("pin_memory", True)
            kwargs.setdefault("num_workers", cpu_count())
        return kwargs


class FrameSampler(Sampler):
    """
    returns SLICE indices
    """
    def __init__(self, N, sequence_length=1, stride=1, shifts=tuple(), shuffle=True):
        super(FrameSampler, self).__init__([0])
        self.base_idx = np.arange(N - sequence_length - sum(shifts) + 1, step=stride)
        self.sequence_length = sequence_length
        self.N = len(self.base_idx)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.base_idx)
        return iter(slice(i, i + self.sequence_length) for i in self.base_idx)

    def __len__(self):
        return self.N


def zip_list(batch_lists):
    return tuple(x for x in zip(*batch_lists))
