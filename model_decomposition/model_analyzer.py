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
from model_decomposition.shape_inference import infer_shapes


class ModelAnalyzer:
    """ Class analyzes an ONNX model and determines how it should be divided into segments. It also decides if the
         segments will stay in ONNX or be converted to TFLite.
    """

    # TODO Idea:
    #  It's not enough to keep a list of ops supported by `onnx2tflite`, because convertibility depends on many other
    #   factors.
    #  1.   Run shape inference. Anything without known shapes must be in ONNX.
    #  2.   For all nodes with known shapes, create a single node model. Use `sympy_data` and create static tensors.
    #        (a lot can be written about the reasons for this in the thesis).
    #       Then try to convert these single node models and perhaps run them to verify correct conversion (with some
    #        atol). If it works, the op will be converted to TFLite.
    #  3.   This is the output of this class. A different one will then split the model into segments accordingly.

    model: onnx.ModelProto

    def __init__(self, model: str | onnx.ModelProto):
        if isinstance(model, str):
            model = onnx.load(model)
        assert isinstance(model, onnx.ModelProto)

        self.model = model

        # Supress the output of the `onnx2tflite`.
        onnx2tflite.src.logger.MIN_OUTPUT_IMPORTANCE = onnx2tflite.src.logger.MessageImportance.ERROR

    def get_nodes_convertible_to_tflite(self) -> list[onnx.NodeProto]:
        """ Analyze the provided ONNX model and determine which nodes can be converted to TFLite and which can't.

        :return: A list of nodes which *can* be converted to TFLite.
        """
        self.model = infer_shapes(self.model)

        convertible_nodes = []
        for node in self.model.graph.node:
            if not node_has_all_shapes_defined(self.model, node):
                # The node will have to stay in ONNX because it uses dynamically sized tensors, which is not supported
                #  by `onnx2tflite`.
                pass

            else:
                # Create a model with just this node and try to convert it to TFLite.
                single_node_model = create_single_node_model(self.model, node)  # TODO Use inferred data too.

                # noinspection PyBroadException
                try:
                    convert_model(single_node_model)

                    # Mark this node as convertible to TFLite.
                    convertible_nodes.append(node)

                except Exception:
                    # It is not possible to convert the node to TFLite.
                    pass

        return convertible_nodes
