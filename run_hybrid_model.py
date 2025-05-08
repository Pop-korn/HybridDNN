#
# Copyright 2025 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from src.model_decomposition.calibration_data_reader import COCOReader
from src.model_format.hybrid_model import HybridModel
from src.model_inference.hdnn_profiler import HDNNProfiler
from src.model_inference.hybrid_model_runner import HybridModelRunner

# Select the hybrid model to run.
hybrid_model_path = 'data/ssd/ssd-12.hdnn'

# Load the hybrid model into memory.
hybrid_model = HybridModel(hybrid_model_path)

# Create the execution provider for the model. The use of the profiler is optional.
profiler = HDNNProfiler()
hybrid_model_runner = HybridModelRunner(hybrid_model, litert_delegate_paths=[], hdnn_profiler=profiler)

# Prepare the input data for the inference.
data_reader = COCOReader()
input_data = [data_reader.get_next() for _ in range(2)]  # Pick the first 2 images.

# Run the model on the input data.
for input_ in input_data:
    output = hybrid_model_runner.run(input_)

# Print out the results of the profiling.
profiler.summarize()
