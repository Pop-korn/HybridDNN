#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

import onnx
import onnx2tflite.src.logger
from onnx2tflite.src.converter.convert import convert_model

from model_decomposition.onnx_model_utils import create_single_node_model, node_has_all_shapes_defined
from model_decomposition.shape_inference import ShapeInference


class ModelAnalyzer:
    """ Class analyzes an ONNX model and determines how it should be divided into segments. It also decides if the
         segments will stay in ONNX or be converted to TFLite.
    """

    model: onnx.ModelProto
    shape_inference: ShapeInference

    def __init__(self, model: str | onnx.ModelProto):
        if isinstance(model, str):
            model = onnx.load(model)
        assert isinstance(model, onnx.ModelProto)

        self.shape_inference = ShapeInference(model)
        self.model = self.shape_inference.run()

        # Supress the output of the `onnx2tflite`.
        onnx2tflite.src.logger.MIN_OUTPUT_IMPORTANCE = onnx2tflite.src.logger.MessageImportance.ERROR

    def get_nodes_convertible_to_tflite(self) -> list[onnx.NodeProto]:
        """ Analyze the provided ONNX model and determine which nodes can be converted to TFLite and which can't.

        :return: A list of nodes which *can* be converted to TFLite.
        """
        convertible_nodes = []
        for node in self.model.graph.node:
            if not node_has_all_shapes_defined(self.model, node):
                # The node will have to stay in ONNX because it uses dynamically sized tensors, which is not supported
                #  by `onnx2tflite`.
                pass

            else:
                # Create a model with just this node and try to convert it to TFLite.
                single_node_model = create_single_node_model(self.model, node, self.shape_inference.get_tensor_data)

                # noinspection PyBroadException
                try:
                    convert_model(single_node_model)

                    # Mark this node as convertible to TFLite.
                    convertible_nodes.append(node)

                except Exception:
                    # It is not possible to convert the node to TFLite.
                    pass

        return convertible_nodes

    # noinspection PyMethodMayBeStatic
    def node_will_be_accelerated_in_tflite(self, node: onnx.NodeProto) -> bool:
        """ Return `True`, if the provided `node` will use HW accelerators after conversion to TFLite. """
        return node.op_type in {'Conv', 'Gemm'}  # TODO Verify and modify.
