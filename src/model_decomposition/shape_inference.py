#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.

import logging

import numpy as np
import onnx
import sympy
from onnx import helper, numpy_helper
from onnx.helper import tensor_dtype_to_np_dtype
from onnx2tflite.src.model_shape_inference import make_dim_param_fixed
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference, get_attribute, get_opset, \
    get_shape_from_sympy_shape, handle_negative_axis, is_literal


# noinspection PyPep8Naming
class ShapeInference:
    """ Class which computes the shapes of internal tensors of an ONNX model, and also computes the output data for some
         nodes when it is statically possible. It extends the ONNX Runtime SymbolicShapeInference to do this.
    """
    model: onnx.ModelProto
    symbolic_shape_inference: SymbolicShapeInference

    def __init__(self, model: onnx.ModelProto, symbolic_dimension_map: dict[str, int] | None = None):
        if symbolic_dimension_map is not None:
            for k, v in symbolic_dimension_map.items():
                make_dim_param_fixed(model.graph, k, v)

        self.symbolic_shape_inference = SymbolicShapeInference(int_max=2 ** 31 - 1, auto_merge=False,
                                                               guess_output_rank=False, verbose=0)
        self.symbolic_shape_inference.dispatcher_['Concat'] = self._infer_Concat
        self.symbolic_shape_inference.dispatcher_['Reshape'] = self._infer_Reshape
        self.symbolic_shape_inference.dispatcher_['Squeeze'] = self._infer_Squeeze
        self.symbolic_shape_inference.dispatcher_['Unsqueeze'] = self._infer_Unsqueeze
        self.model = model

    # noinspection PyProtectedMember
    def _try_get_value(self, node: onnx.NodeProto, input_index: int) -> np.ndarray | None:
        """ Return the data for the input tensor on index `input_index` for the given node, if it exists. Otherwise,
             return `None`.

        :param node: Node to get the input data for.
        :param input_index: Index to the `node.input`.
        :return: The data of the input, or `None`.
        """
        return self.symbolic_shape_inference._try_get_value(node, input_index)

    # noinspection PyProtectedMember
    def _get_shape(self, node: onnx.NodeProto, input_index: int) -> list[int] | None:
        """ Return the shape for the input tensor on index `input_index` for the given node, if it exists. Otherwise,
             return `None`.

        :param node: Node to get the input shape for.
        :param input_index: Index to the `node.input`.
        :return: The shape of the input, or `None`.
        """
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

    @property
    def sympy_data_(self) -> dict[str, np.ndarray | list]:
        return self.symbolic_shape_inference.sympy_data_

    @property
    def initializers_(self) -> dict[str, onnx.TensorProto]:
        return self.symbolic_shape_inference.initializers_

    def _infer_Reshape(self, node):  # noqa: N802
        """ Custom dispatcher for the `Reshape` operator. It extends the original one by ONNX Runtime, to solve data
             inference issues.

            :param node: Node to infer the shapes and data for.
        """

        # ---- START OF CODE TAKEN FROM ONNX RUNTIME ----
        shape_value = self._try_get_value(node, 1)
        vi = self.known_vi_[node.output[0]]
        if shape_value is None:
            shape_shape = self._get_shape(node, 1)
            assert len(shape_shape) == 1
            shape_rank = shape_shape[0]
            assert is_literal(shape_rank)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(self.symbolic_shape_inference._new_symbolic_shape(shape_rank, node)),
                )
            )
        else:
            input_sympy_shape = self.symbolic_shape_inference._get_sympy_shape(node, 0)
            total = 1
            for d in input_sympy_shape:
                total = total * d
            new_sympy_shape = []
            deferred_dim_idx = -1
            non_deferred_size = 1
            for i, d in enumerate(shape_value):
                if type(d) == sympy.Symbol:
                    new_sympy_shape.append(d)
                elif d == 0:
                    new_sympy_shape.append(input_sympy_shape[i])
                    non_deferred_size = non_deferred_size * input_sympy_shape[i]
                else:
                    new_sympy_shape.append(d)
                if d == -1:
                    deferred_dim_idx = i
                elif d != 0:
                    non_deferred_size = non_deferred_size * d

            assert new_sympy_shape.count(-1) < 2
            if -1 in new_sympy_shape:
                new_dim = total // non_deferred_size
                new_sympy_shape[deferred_dim_idx] = new_dim

            self.symbolic_shape_inference._update_computed_dims(new_sympy_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )
            # ---- END OF CODE TAKEN FROM ONNX RUNTIME ----
            
            # Try to infer the output data.
            # noinspection PyBroadException
            try:
                input_data = self._try_get_value(node, 0)
                if input_data is not None:
                    np_type = tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
                    output_data = np.asarray(input_data).astype(np_type).reshape(new_sympy_shape)
                    self.symbolic_shape_inference.sympy_data_[node.output[0]] = list(output_data)
            except Exception:
                pass  # Failed to inter the data. No action needed.

    def _infer_Unsqueeze(self, node):  # noqa: N802
        """ Custom dispatcher for the `Unsqueeze` operator. It extends the original one by ONNX Runtime, to solve data
             inference issues.

            :param node: Node to infer the shapes and data for.
        """
        
        # ---- START OF CODE TAKEN FROM ONNX RUNTIME ----
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
        # ---- END OF CODE TAKEN FROM ONNX RUNTIME ----

        # Try to infer the output data.
        # noinspection PyBroadException
        try:
            input_data = self._try_get_value(node, 0)
            if input_data is not None:
                np_type = tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
                output_data = np.asarray(input_data).astype(np_type).reshape(output_shape)
                self.symbolic_shape_inference.sympy_data_[node.output[0]] = list(output_data)
        except Exception:
            pass  # Failed to inter the data. No action needed.

    def _infer_Squeeze(self, node):  # noqa: N802
        """ Custom dispatcher for the `Squeeze` operator. It extends the original one by ONNX Runtime, to solve data
             inference issues.

            :param node: Node to infer the shapes and data for.
        """

        # ---- START OF CODE TAKEN FROM ONNX RUNTIME ----
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            # assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            # assert get_attribute(node, "axes") is None

        if axes is None:
            # No axes have been provided (neither via attribute nor via input).
            # In this case the 'Shape' op should remove all axis with dimension 1.
            # For symbolic dimensions we guess they are !=1.
            output_shape = [s for s in input_shape if s != 1]
            # if self.verbose_ > 0:
            #     symbolic_dimensions = [s for s in input_shape if type(s) != int]  # noqa: E721
            #     if len(symbolic_dimensions) > 0:
            #         logger.debug(
            #             f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
            #             f"Assuming the following dimensions are never equal to 1: {symbolic_dimensions}"
            #         )
        else:
            axes = [handle_negative_axis(a, len(input_shape)) for a in axes]
            output_shape = []
            for i in range(len(input_shape)):
                if i not in axes:
                    output_shape.append(input_shape[i])
                else:
                    assert input_shape[i] == 1 or type(input_shape[i]) != int  # noqa: E721
                    # if self.verbose_ > 0 and type(input_shape[i]) != int:  # noqa: E721
                    #     logger.debug(
                    #         f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                    #         f"Assuming the dimension '{input_shape[i]}' at index {i} of the input to be equal to 1."
                    #     )

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )
        # ---- END OF CODE TAKEN FROM ONNX RUNTIME ----

        # Try to infer the output data.
        # noinspection PyBroadException
        try:
            input_data = self._try_get_value(node, 0)
            if input_data is not None:
                np_type = tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
                output_data = np.asarray(input_data).astype(np_type).reshape(output_shape)
                self.symbolic_shape_inference.sympy_data_[node.output[0]] = list(output_data)
        except Exception:
            pass  # Failed to inter the data. No action needed.

    def _infer_Concat(self, node):  # noqa: N802
        """ Custom dispatcher for the `Concat` operator. It extends the original one by ONNX Runtime, to solve data
             inference issues.

            :param node: Node to infer the shapes and data for.
        """
        # Try to infer the data.
        # noinspection PyBroadException
        try:
            if any([i in self.sympy_data_ or i in self.initializers_ for i in node.input]):
                values = self.symbolic_shape_inference._get_int_or_float_values(node)
                if all([v is not None for v in values]):
                    assert get_attribute(node, "axis") == 0
                    self.sympy_data_[node.output[0]] = []
                    for i in range(len(node.input)):
                        value = values[i]
                        if isinstance(value, list):
                            self.sympy_data_[node.output[0]].extend(value)
                        else:
                            self.sympy_data_[node.output[0]].append(value)
        except Exception:
            pass  # Failed to inter the data. No action needed.

        # ---- START OF CODE TAKEN FROM ONNX RUNTIME ----
        sympy_shape = self.symbolic_shape_inference._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis"), len(sympy_shape))
        for i_idx in range(1, len(node.input)):
            input_shape = self.symbolic_shape_inference._get_sympy_shape(node, i_idx)
            if input_shape:
                sympy_shape[axis] = sympy_shape[axis] + input_shape[axis]
        self.symbolic_shape_inference._update_computed_dims(sympy_shape)
        # merge symbolic dims for non-concat axes
        for d in range(len(sympy_shape)):
            if d == axis:
                continue
            dims = [self._get_shape(node, i_idx)[d] for i_idx in range(len(node.input)) if self._get_shape(node, i_idx)]
            if all([d == dims[0] for d in dims]):
                continue
            merged = self.symbolic_shape_inference._merge_symbols(dims)
            if type(merged) == str:  # noqa: E721
                sympy_shape[d] = self.symbolic_shape_inference.symbolic_dims_[merged] if merged else None
            else:
                sympy_shape[d] = merged
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )
        # ---- END OF CODE TAKEN FROM ONNX RUNTIME ----

    # noinspection PyProtectedMember
    def _infer_shapes(self):
        """ Infer the shapes of internal tensors of `self.model`. """
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
            logging.warning('Incomplete shape inference')

        self.model = self.symbolic_shape_inference.out_mp_
        return self.model

    def run(self) -> onnx.ModelProto:
        """ MAIN ENTRY POINT. Run the shape inference and return the ONNX model with inferred shapes. """
        return self._infer_shapes()

    def get_tensor_data(self, tensor_name: str) -> np.ndarray | None:
        """ Get the data for the tensor with the given name, or `None` if the data is not available.

        :param tensor_name: Name of the tensor to get the data for.
        :return: The data of the tensor, or `None`.
        """
        if tensor_name in self.symbolic_shape_inference.sympy_data_ or tensor_name in self.symbolic_shape_inference.initializers_:
            data = self.symbolic_shape_inference.sympy_data_[
                tensor_name] if tensor_name in self.symbolic_shape_inference.sympy_data_ else numpy_helper.to_array(
                self.symbolic_shape_inference.initializers_[tensor_name])

            return np.asarray(data)
        return None
