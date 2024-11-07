#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.

import logging

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference, get_opset


class ShapeInference:
    model: onnx.ModelProto
    symbolic_shape_inference: SymbolicShapeInference

    def __init__(self, model: onnx.ModelProto):
        self.symbolic_shape_inference = SymbolicShapeInference(int_max=2 ** 31 - 1, auto_merge=False,
                                                               guess_output_rank=False, verbose=0)
        self.model = model

    # noinspection PyProtectedMember
    def _infer_shapes(self):
        onnx_opset = get_opset(self.model)
        if (not onnx_opset) or onnx_opset < 7:
            logging.warning("Only support models of onnx opset 7 and above.")
            return None

        all_shapes_inferred = False
        self.symbolic_shape_inference._preprocess(self.model)
        while self.symbolic_shape_inference.run_:
            all_shapes_inferred = self.symbolic_shape_inference._infer_impl()
        self.symbolic_shape_inference._update_output_from_vi()
        if not all_shapes_inferred:
            onnx.save_model(self.symbolic_shape_inference.out_mp_, "sym_shape_infer_temp.onnx",
                            save_as_external_data=True)
            raise Exception("Incomplete symbolic shape inference")

        self.model = self.symbolic_shape_inference.out_mp_
        return self.model

    def run(self) -> onnx.ModelProto:
        return self._infer_shapes()

    def get_tensor_data(self, tensor_name: str) -> np.ndarray | None:
        if tensor_name in self.symbolic_shape_inference.sympy_data_ or tensor_name in self.symbolic_shape_inference.initializers_:
            data = self.symbolic_shape_inference.sympy_data_[
                tensor_name] if tensor_name in self.symbolic_shape_inference.sympy_data_ else numpy_helper.to_array(
                self.symbolic_shape_inference.initializers_[tensor_name])

            return np.asarray(data)
        return None
