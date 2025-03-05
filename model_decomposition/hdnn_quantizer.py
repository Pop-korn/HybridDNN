#
# Copyright 2025 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from collections import defaultdict
from typing import Iterable

import numpy as np
import onnx
from onnx2quant.qdq_quantization import CalibrationDataReader, QDQQuantizer, QuantizationConfig

from model_inference.hybrid_model_runner import ONNXRunner


class EvolvingDataReader(CalibrationDataReader):
    all_data = dict[str, list[np.ndarray]]  # All input data for all known tensor names.
    current_input_data_iter = Iterable[dict[str, np.ndarray]]
    num_iterations: int

    def __init__(self, initial_data_reader: CalibrationDataReader):
        super().__init__()
        self.all_data = defaultdict(list)
        first_input_data = []
        self.num_iterations = 0
        while (data := initial_data_reader.get_next()) is not None:
            self.num_iterations += 1
            first_input_data.append(data)
            for k, v in data.items():
                self.all_data[k].append(v)

        self.current_input_data_iter = iter(first_input_data)

    def compute_new_calibration_data(self, onnx_model: onnx.ModelProto, input_names: list[str]):
        onnx_runner = ONNXRunner(onnx_model.SerializeToString())
        for input_dict in self._get_input_data_for_names(input_names):
            output_dict = onnx_runner.run(input_dict)
            for name, data in output_dict.items():
                self.all_data[name].append(data)


    def _get_input_data_for_names(self, input_names: list[str]) -> list[dict[str, np.ndarray]]:
        new_input_data = []
        for iteration in range(self.num_iterations):
            new_input_data.append(
                {
                    input_name: all_data_for_input[iteration]
                    for input_name, all_data_for_input in self.all_data.items() if input_name in input_names
                }
            )

        return new_input_data

    def prepare_input_data(self, input_names: list[str]):
        self.current_input_data_iter = iter(self._get_input_data_for_names(input_names))

    def get_next(self) -> dict[str, np.ndarray]:
        return next(self.current_input_data_iter, None)

class HDNNQuantizer:
    evolving_data_reader: EvolvingDataReader

    def __init__(self, input_data_reader: CalibrationDataReader):
        self.evolving_data_reader = EvolvingDataReader(input_data_reader)

    def compute_new_calibration_date(self, onnx_model: onnx.ModelProto, input_names: list[str]):
        self.evolving_data_reader.compute_new_calibration_data(onnx_model, input_names)

    def quantize(self, onnx_model: onnx.ModelProto, inputs: list[str]) -> onnx.ModelProto:
        # Quantize the ONNX model via `onnx2quant`.

        self.evolving_data_reader.prepare_input_data(inputs)

        q_config = QuantizationConfig(self.evolving_data_reader)
        return QDQQuantizer().quantize_model(onnx_model, q_config)
