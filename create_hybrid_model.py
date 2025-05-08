#
# Copyright 2025 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

import onnx

from src.model_decomposition.calibration_data_reader import COCOReader
from src.model_decomposition.model_decomposer import DecompositionStrategy, ModelDecomposer

# Select the working directory and the input ONNX model which will be turned into a hybrid model.
directory = 'data/ssd/'
model_name = 'ssd-12'
onnx_model_name = directory + model_name + '.onnx'
hdnn_model_name = directory + model_name + '.hdnn'

# Pick the decomposition strategy. `RATIONAL` yields a less fragmented model which results in faster inference.
strategy = DecompositionStrategy.RATIONAL

# Create a calibration data reader, which will provide input data for quantization.
calibration_data_reader = COCOReader()

# Increase the op-set version of the ONNX model, to make it compatible with third party tools.
onnx_model = onnx.load(onnx_model_name)
onnx_model = onnx.version_converter.convert_version(onnx_model, 13)

# Create the hybrid model.
decomposer = ModelDecomposer(onnx_model, calibration_data_reader)
hybrid_model = decomposer.create_hybrid_model(strategy)

# Print the internal composition of the hybrid model.
print(hybrid_model)

# Save the internal segments of the model for analysis.
hybrid_model.save_segments(directory + (
    'submodels_rational' if strategy == DecompositionStrategy.RATIONAL else 'submodels_naive'
))

# Save the full hybrid model to a `.hdnn` file.
hybrid_model.store(hdnn_model_name)
