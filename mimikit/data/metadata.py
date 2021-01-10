import pandas as pd
import numpy as np


def ssts(item, axis=0):
    arr = np.atleast_2d(item[['start', 'stop']].values)
    slices = map(lambda a: (*[slice(0, None)] * axis, slice(a[0], a[1]), ), arr)
    return slices


class Metadata(pd.DataFrame):
    """
    subclass of ``pandas.DataFrame`` to store a slice structure. This is handy when concatenating arrays

    Instances of ``Metadata`` created by the constructors mentioned below will automatically have
    three columns : "start", "stop", "duration".

    :no-inherited-members:
    """

    @property
    def _constructor(self):
        return Metadata

    @staticmethod
    def _validate(obj):
        if 'start' not in obj.columns or "stop" not in obj.columns:
            raise ValueError("Must have 'start' and 'stop' columns.")

    # PROPERTIES

    @property
    def first_start(self):
        return self.start.min()

    @property
    def last_stop(self):
        return self.stop.max()

    @property
    def span(self):
        return self.last_stop - self.first_start

    @property
    def cumdur(self):
        return np.cumsum(self["duration"].values)

    @property
    def all_indices(self):
        return np.array([i for ev in self.events for i in range(ev.start, ev.stop)])

    def slices(self, time_axis=0):
        """
        This is the back-end core of a score. This method efficiently returns the indexing objects
        necessary to communicate with n-dimensional data.

        @return: an array of slices where slice_i corresponds to the row/Event_i in the DataFrame
        """
        return ssts(self, time_axis)

    @property
    def durations_(self):
        return self.stop.values - self.start.values

    @property
    def events(self):
        return self.itertuples(name="Event", index=True)

    # UPDATING METHOD

    def make_contiguous(self):
        self.reset_index(drop=True, inplace=True)
        cumdur = self.cumdur
        self.loc[:, "start"] = np.r_[0, cumdur[:-1]] if cumdur[0] != 0 else cumdur[:-1]
        self.loc[:, "stop"] = cumdur
        return self

    # Sanity Checks

    def is_consistent(self):
        return (self["duration"].values == (self.stop.values - self.start.values)).all()

    def is_contiguous(self):
        return (self.start.values[1:] == self.stop.values[:-1]).all()

    @staticmethod
    def from_start_stop(starts, stops, durations):
        return Metadata((starts, stops, durations), index=["start", "stop", "duration"]).T

    @staticmethod
    def from_stop(stop):
        """
        integers in `stops` correspond to the prev[stop] and next[start] values.
        `stops` must contain the last index ! and it can begin with 0, but doesn't have to...
        """
        stop = np.asarray(stop)
        if stop[0] == 0:
            starts = np.r_[0, stop[:-1]]
            stop = stop[1:]
        else:
            starts = np.r_[0, stop[:-1]]
        durations = stop - starts
        return Metadata.from_start_stop(starts, stop, durations)

    @staticmethod
    def from_duration(duration):
        stops = np.cumsum(duration)
        starts = np.r_[0, stops[:-1]]
        return Metadata.from_start_stop(starts, stops, duration)

    @staticmethod
    def from_data(sequence, time_axis=1):
        duration = np.array([x.shape[time_axis] for x in sequence])
        return Metadata.from_duration(duration)

    @staticmethod
    def from_frame_definition(total_duration, frame_length, stride=1, butlasts=0):
        starts = np.arange(total_duration - frame_length - butlasts + 1, step=stride)
        durations = frame_length + np.zeros_like(starts, dtype=np.int)
        stops = starts + durations
        return Metadata.from_start_stop(starts, stops, durations)


@pd.api.extensions.register_dataframe_accessor("soft_q")
class SoftQueryAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def or_(self, **kwargs):
        series = False
        for col_name, func in kwargs.items():
            series = series | func(self._df[col_name])
        return series

    def and_(self, **kwargs):
        series = True
        for col_name, func in kwargs.items():
            series = series & func(self._df[col_name])
        return series


@pd.api.extensions.register_dataframe_accessor("hard_q")
class HardQueryAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def or_(self, **kwargs):
        series = self._df.soft_q.or_(**kwargs)
        return self._df[series]

    def and_(self, **kwargs):
        series = self._df.soft_q.and_(**kwargs)
        return self._df[series]
