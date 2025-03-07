#
# Copyright 2024-2025 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from abc import ABC, abstractmethod

import numpy as np
from onnxruntime import InferenceSession
from tflite_runtime.interpreter import Interpreter as LiteRTInterpreter

from model_format.hybrid_model import HybridModel, ModelFormat


class ModelRunner(ABC):
    @abstractmethod
    def __init__(self, model_raw_data: bytes):
        raise NotImplementedError

    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError


class LiteRTRunner(ModelRunner):
    litert_interpreter: LiteRTInterpreter

    input_details: list[dict[str, any]]
    output_details: list[dict[str, any]]

    def __init__(self, model_raw_data: bytes):
        self.litert_interpreter = LiteRTInterpreter(model_content=model_raw_data)
        self.litert_interpreter.allocate_tensors()

        self.input_details = self.litert_interpreter.get_input_details()
        self.output_details = self.litert_interpreter.get_output_details()

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Set the input data.
        for input_detail in self.input_details:
            self.litert_interpreter.set_tensor(input_detail['index'], inputs[input_detail['name']])

        self.litert_interpreter.invoke()

        # Get the output data.
        outputs = {
            output_detail['name']: self.litert_interpreter.get_tensor(output_detail['index'])
            for output_detail in self.output_details
        }

        return outputs


class ONNXRunner(ModelRunner):
    onnx_inference_session: InferenceSession

    output_names: list[str]  # Name of the output tensors of the segment in the correct order.

    def __init__(self, model_raw_data: bytes):
        self.onnx_inference_session = InferenceSession(model_raw_data)
        self.output_names = [output_vi.name for output_vi in self.onnx_inference_session.get_outputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output_tensors = self.onnx_inference_session.run(None, inputs)
        outputs = dict(zip(self.output_names, output_tensors))
        return outputs


class HybridModelRunner:
    hybrid_model: HybridModel

    # A list of ModelRunners for every segment. The order of the runners matches the order of the segments in the
    #  `hybrid_model.segments`.
    segment_runners: list[ModelRunner]

    def __init__(self, hybrid_model: HybridModel):
        self.hybrid_model = hybrid_model
        self.segment_runners = []

        for segment in self.hybrid_model.model_segments:
            if segment.format == ModelFormat.LiteRT:
                self.segment_runners.append(LiteRTRunner(segment.raw_data))

            elif segment.format == ModelFormat.ONNX:
                self.segment_runners.append(ONNXRunner(segment.raw_data))

            else:
                raise ValueError(f'HybridModelInterpreter: invalid segment format `{segment.format}`.')

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str: np.ndarray]:
        """ Run the hybrid model with the provided `inputs` and return the produced output tensors. """
        known_tensors = inputs.copy()

        for segment, segment_runner in zip(self.hybrid_model.model_segments, self.segment_runners):
            # TODO This significantly slows down the inference. Verify statically in the `__init__()` method that this
            #  will not cause problems.
            # for input_ in segment.inputs:
            #     if input_ not in known_tensors.keys():
            #         raise KeyError(f'HybridModelRunner.run(): Invalid hybrid model. Segment `{segment.file_name}` '
            #                        f'requires the input tensor `{input_}`, which is not a model input nor an output of '
            #                        'a previous segment.')


            inputs = {name: data for name, data in known_tensors.items() if name in segment.inputs}
            outputs = segment_runner.run(inputs)
            for k, output in outputs.items():
                if output.dtype == np.longlong:
                    output = output.astype('int64')
                known_tensors[k] = output

        # Return the data of the model outputs.
        return {name: data for name, data in known_tensors.items() if name in self.hybrid_model.outputs}
