#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#
import logging

import numpy as np
import onnx
import onnx2tflite.src.logger
from onnx.helper import tensor_dtype_to_np_dtype
from onnx2tflite.src.converter.convert import convert_model
from onnxruntime.tools.symbolic_shape_infer import get_shape_from_value_info

from model_decomposition.onnx_model_utils import create_single_node_model, node_has_all_shapes_defined
from model_decomposition.shape_inference import ShapeInference


class ModelAnalyzer:
    """ Class analyzes an ONNX model and determines how it should be divided into segments. It also decides if the
         segments will stay in ONNX or be converted to LiteRT.
    """

    model: onnx.ModelProto
    shape_inference: ShapeInference

    def __init__(self, model: str | onnx.ModelProto):
        if isinstance(model, str):
            model = onnx.load(model)
        assert isinstance(model, onnx.ModelProto)

        self.shape_inference = ShapeInference(model)
        self.model = self.shape_inference.run()

        # Increase the opset_version to the minimum required by ONNXRuntime
        self.model = onnx.version_converter.convert_version(self.model, 13)

        # Remove ONNX nodes for which the output data is known.
        self.remove_nodes_with_known_outputs()

        # Run shape (and data) inference on the final model.
        self.shape_inference = ShapeInference(self.model)
        self.model = self.shape_inference.run()

        # Supress the output of the `onnx2tflite`.
        onnx2tflite.src.logger.MIN_OUTPUT_IMPORTANCE = onnx2tflite.src.logger.MessageImportance.ERROR

    def get_nodes_convertible_to_litert(self) -> list[onnx.NodeProto]:
        """ Analyze the provided ONNX model and determine which nodes can be converted to LiteRT and which can't.

        :return: A list of nodes which *can* be converted to LiteRT.
        """
        convertible_nodes = []
        for node in self.model.graph.node:
            if not node_has_all_shapes_defined(self.model, node):
                # The node will have to stay in ONNX because it uses dynamically sized tensors, which is not supported
                #  by `onnx2tflite`.
                pass

            else:
                # Create a model with just this node and try to convert it to LiteRT.
                single_node_model = create_single_node_model(self.model, node, self.shape_inference.get_tensor_data)

                # noinspection PyBroadException
                try:
                    convert_model(single_node_model)

                    # Mark this node as convertible to LiteRT.
                    convertible_nodes.append(node)

                except Exception:
                    # It is not possible to convert the node to LiteRT.
                    pass

        return convertible_nodes

    # noinspection PyMethodMayBeStatic
    def node_will_be_accelerated_in_litert(self, node: onnx.NodeProto) -> bool:
        """ Return `True`, if the provided `node` will use HW accelerators after conversion to LiteRT. """
        return node.op_type in {'Conv', 'Gemm'}  # TODO Verify and modify.

    def remove_nodes_with_known_outputs(self):
        nodes_to_remove = []
        for node in self.model.graph.node:
            if all(self.shape_inference.symbolic_shape_inference.sympy_data_.get(o, None) is not None
                   for o in node.output):
                try:
                    initializers_to_add = []
                    for o in node.output:
                        data = self.shape_inference.symbolic_shape_inference.sympy_data_[o]
                        output_vi = self.shape_inference.symbolic_shape_inference.known_vi_[o]
                        np_type = tensor_dtype_to_np_dtype(output_vi.type.tensor_type.elem_type)
                        data = np.asarray(data, np_type)
                        if list(data.shape) != get_shape_from_value_info(output_vi):
                            logging.warning(f'The value info shape, and inferred data shape differ for `{o}`.')
                            raise RuntimeError
                        static_tensor = onnx.numpy_helper.from_array(data, o)
                        initializers_to_add.append(static_tensor)

                    for initializer in initializers_to_add:
                        self.model.graph.initializer.append(initializer)
                    nodes_to_remove.append(node)
                except Exception:
                    pass

        for node in nodes_to_remove:
            self.model.graph.node.remove(node)
