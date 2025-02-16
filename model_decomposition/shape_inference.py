#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.

import logging

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.helper import tensor_dtype_to_np_dtype
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference, get_attribute, get_opset, \
    handle_negative_axis


# noinspection PyPep8Naming
class ShapeInference:
    model: onnx.ModelProto
    symbolic_shape_inference: SymbolicShapeInference

    def __init__(self, model: onnx.ModelProto):
        self.symbolic_shape_inference = SymbolicShapeInference(int_max=2 ** 31 - 1, auto_merge=False,
                                                               guess_output_rank=False, verbose=0)
        self.symbolic_shape_inference.dispatcher_['Unsqueeze'] = self._infer_Unsqueeze
        self.model = model

    # noinspection PyProtectedMember
    def _try_get_value(self, node: onnx.NodeProto, input_index: int) -> np.ndarray | None:
        return self.symbolic_shape_inference._try_get_value(node, input_index)

    # noinspection PyProtectedMember
    def _get_shape(self, node: onnx.NodeProto, input_index: int) -> list[int] | None:
        try:
            return self.symbolic_shape_inference._get_shape(node, input_index)
        except AssertionError:
            return None

    @property
    def out_mp_(self) -> onnx.ModelProto:
        return self.symbolic_shape_inference.out_mp_

    @property
    def known_vi_(self) -> dict[str, onnx.ValueInfoProto]:
        return self.symbolic_shape_inference.known_vi_

    def _infer_Unsqueeze(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        output_rank = len(input_shape) + len(axes)
        axes = [handle_negative_axis(a, output_rank) for a in axes]

        input_axis = 0
        output_shape = []
        for i in range(output_rank):
            if i in axes:
                output_shape.append(1)
            else:
                output_shape.append(input_shape[input_axis])
                input_axis += 1

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

        # Try to infer the output data.
        # noinspection PyBroadException
        try:
            input_data = self._try_get_value(node, 0)
            if input_data is not None:
                np_type = tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
                output_data = np.asarray(input_data).astype(np_type).reshape(output_shape)
                self.symbolic_shape_inference.sympy_data_[node.output[0]] = list(output_data)
        except Exception as e:
            pass  # Failed to inter the data. No action needed.

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
