#
# Copyright 2024-2025 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

import numpy as np
from abc import ABC, abstractmethod
from onnxruntime import InferenceSession

from src.model_inference.hdnn_profiler import HDNNProfiler

# If executed on i.MX platform, there is no tensorflow module. And typically the intention is to use the tflite python
# interpreter available in tflite_runtime
try:
    import tensorflow.lite as tflite
except ModuleNotFoundError:
    import tflite_runtime.interpreter as tflite

from src.model_format.hybrid_model import HybridModel, ModelFormat


class ModelRunner(ABC):
    """ Abstract interface class to run inference on a DNN model. """

    @abstractmethod
    def __init__(self, model_raw_data: bytes):
        raise NotImplementedError

    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError


class LiteRTRunner(ModelRunner):
    """ Helper class to run inference on a LiteRT model. """

    # The interpreter should support WH accelerators if available.
    litert_interpreter: tflite.Interpreter

    input_details: list[dict[str, any]]
    output_details: list[dict[str, any]]

    def __init__(
            self,
            model_raw_data: bytes,
            delegate_paths: list[str],
            model_name: str = "unnamed_litert_model",
            profiler: HDNNProfiler | None = None
    ):
        self.model_name = model_name
        self.profiler = profiler

        self.litert_interpreter = tflite.Interpreter(
            model_content=model_raw_data,
            experimental_delegates=[tflite.load_delegate(delegate_path) for delegate_path in delegate_paths]
        )
        self.litert_interpreter.allocate_tensors()

        self.input_details = self.litert_interpreter.get_input_details()
        self.output_details = self.litert_interpreter.get_output_details()

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Run inference of the LiteRT model with the provided data, and return the outputs.

        :param inputs: Dictionary mapping names of input tensors to their data.
        :return: Dictionary mapping names of output tensors to their data.
        """
        # Set the input data.
        for input_detail in self.input_details:
            self.litert_interpreter.set_tensor(input_detail['index'], inputs[input_detail['name']])

        if self.profiler is None:
            self.litert_interpreter.invoke()
        else:
            with self.profiler.time(self.model_name):
                self.litert_interpreter.invoke()

        # Get the output data.
        outputs = {
            output_detail['name']: self.litert_interpreter.get_tensor(output_detail['index'])
            for output_detail in self.output_details
        }

        return outputs


class ONNXRunner(ModelRunner):
    """ Helper class to run inference on an ONNX model. """

    onnx_inference_session: InferenceSession

    output_names: list[str]  # Name of the output tensors of the segment in the correct order.

    def __init__(
            self,
            model_raw_data: bytes,
            model_name: str = "unnamed_onnx_model",
            profiler: HDNNProfiler | None = None
    ):
        self.model_name = model_name
        self.profiler = profiler
        self.onnx_inference_session = InferenceSession(model_raw_data)
        self.output_names = [output_vi.name for output_vi in self.onnx_inference_session.get_outputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Run inference of the LiteRT model with the provided data, and return the outputs.

        :param inputs: Dictionary mapping names of input tensors to their data.
        :return: Dictionary mapping names of output tensors to their data.
        """
        if self.profiler is None:
            output_tensors = self.onnx_inference_session.run(None, inputs)
        else:
            with self.profiler.time(self.model_name):
                output_tensors = self.onnx_inference_session.run(None, inputs)

        outputs = dict(zip(self.output_names, output_tensors))
        return outputs


class HybridModelRunner:
    """ Class to run efficient inference on a HybridModel using existing ONNX and LiteRT inference providers. """

    hybrid_model: HybridModel

    # A list of ModelRunners for every segment. The order of the runners matches the order of the segments in the
    #  `hybrid_model.segments`.
    segment_runners: list[ModelRunner]

    def __init__(
            self,
            hybrid_model: HybridModel,
            litert_delegate_paths: list[str] | None = None,
            hdnn_profiler: HDNNProfiler | None = None
    ):
        self.hybrid_model = hybrid_model
        self.segment_runners = []

        if litert_delegate_paths is None:
            litert_delegate_paths = ['/usr/lib/libvx_delegate.so']  # Delegate on i.MX 8M Plus

        for segment in self.hybrid_model.model_segments:
            if segment.format == ModelFormat.LiteRT:
                self.segment_runners.append(
                    LiteRTRunner(segment.raw_data, litert_delegate_paths, segment.file_name, hdnn_profiler)
                )

            elif segment.format == ModelFormat.ONNX:
                self.segment_runners.append(
                    ONNXRunner(segment.raw_data, segment.file_name, hdnn_profiler)
                )

            else:
                raise ValueError(f'HybridModelInterpreter: invalid segment format `{segment.format}`.')

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str: np.ndarray]:
        """ Run the hybrid model with the provided `inputs` and return the produced output tensors.

        :param inputs: Dictionary mapping names of input tensors to their data.
        :return: Dictionary mapping names of output tensors to their data.
        """
        known_tensors = inputs.copy()

        for segment, segment_runner in zip(self.hybrid_model.model_segments, self.segment_runners):
            inputs = {name: data for name, data in known_tensors.items() if name in segment.inputs}
            outputs = segment_runner.run(inputs)
            for k, output in outputs.items():
                if output.dtype == np.longlong:
                    output = output.astype('int64')
                known_tensors[k] = output

        # Return the data of the model outputs.
        return {name: data for name, data in known_tensors.items() if name in self.hybrid_model.outputs}
