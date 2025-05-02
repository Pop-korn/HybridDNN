#
# Copyright 2025 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

import numpy as np
import time
from collections import defaultdict


class TimeRecorder:
    recorded_times: list[float]

    def __init__(self):
        self.recorded_times = []

    def record(self, time_: float):
        self.recorded_times.append(time_)


class Timer:
    start_time: float
    recorder: TimeRecorder

    def __init__(self, recorder: TimeRecorder):
        self.recorder = recorder

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.recorder.record(time.time() - self.start_time)


class HDNNProfiler:
    """ Class enables the profiling of a Hybrid Model. It is used by the `HybridModelRunner` to record the inference
         times for all the internal segments which make up the model.

        The profiling introduces a slight overhead, so it shouldn't be used once the model is deployed, or when the
         inference time of the entire model is being measured.


        Usage:
        ```
            hdnn_profiler = HDNNProfiler()
            hdnn_runner = HybridModelRunner(..., hdnn_profiler=hdnn_profiler)
            ...
            hdnn_runner.run(...)
            ...
            hdnn_profiler.summarize()
        ```
    """
    recorders: defaultdict[str, TimeRecorder]

    def __init__(self):
        self.recorders = defaultdict(TimeRecorder)

    def time(self, segment_name: str) -> Timer:
        """ Create a `Timer` object for a model segment with the given name.

            Usage:
            ```
                with hdnn_profiler.time('segment name'):
                    <run the segment>
            ```
        """
        return Timer(self.recorders[segment_name])

    def summarize(self):
        """ Print a summary of the measured times. """
        print('Profiling summary:')
        average_segment_times = []
        for segment, recorder in self.recorders.items():
            data = np.asarray(recorder.recorded_times)
            average_segment_times.append(data.mean())
            print(
                f'\t`{segment}`: '
                f'\n\t\tavg = {data.mean()}'
                f'\n\t\tmax = {data.max()}'
                f'\n\t\tmin = {data.min()}',
                end=''
            )
            if len(recorder.recorded_times) > 1:
                print(f'\n\t\tavg (without 1st run) = {data[1:].mean()}')
            else:
                print()

        print(f'Sum of the average times of all segments: {sum(average_segment_times)}')

