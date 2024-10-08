#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#
from dataclasses import dataclass

import onnx

from model_decomposition.model_analyzer import ModelAnalyzer
from model_format.hybrid_model import ModelFormat


@dataclass
class NodeGroup:
    nodes: list[onnx.NodeProto]
    tensors: list[str]  # List of names of tensors which are produced by this group of nodes.
    format: ModelFormat | None = None


class ModelDecomposer:
    """ Class divides the input ONNX model into sections. Some of them get converted to TFLite and some stay in ONNX.
         These sections are then combined into a hybrid model (hdnn).
    """

    model: onnx.ModelProto

    def __init__(self, model: onnx.ModelProto | str):
        if isinstance(model, str):
            self.model = onnx.load(model)
        else:
            self.model = model

    def create_hybrid_model(self):  # TODO Select decomposition strategy.
        analyzer = ModelAnalyzer(self.model)
        convertible_nodes = analyzer.get_nodes_convertible_to_tflite()

        groups = [
            NodeGroup(
                nodes=[],
                tensors=[input_.name for input_ in self.model.graph.input] +  # Global model inputs.
                        [t.name for t in self.model.graph.initializer]  # Static tensors.
            )
        ]

        def _get_group_for_tensor(tensor_name: str) -> int:
            for idx, group_ in enumerate(groups):
                if tensor_name in group_.tensors:
                    return idx

            raise KeyError

        def _get_last_group_this_node_depends_on(node_: onnx.NodeProto) -> int:
            last_group = 0
            for input_ in node_.input:
                last_group = max(_get_group_for_tensor(input_), last_group)

            return last_group

        def _node_format(node_: onnx.NodeProto) -> ModelFormat:
            return ModelFormat.TFLite if node_ in convertible_nodes else ModelFormat.ONNX

        current_group = 0
        for node in self.model.graph.node:
            # noinspection PySimplifyBooleanCheck
            if groups[0].nodes == []:
                # This is the first node. It will always be in the 0th group.
                groups[0].nodes.append(node)
                groups[0].tensors.extend(node.output)
                groups[0].format = _node_format(node)
                continue

            # -- Not the first node. --

            previous_group = _get_last_group_this_node_depends_on(node)

            # Add this node to the earliest possible group, starting with `previous_group`. But it has to be a group of
            #  the same format.
            node_added = False
            for group in groups[previous_group:]:
                if group.format == _node_format(node):
                    group.nodes.append(node)
                    group.tensors.extend(node.output)
                    node_added = True

            if not node_added:
                # There is no existing group suitable for `node`. Create a new group and add the `node` to it.
                # This can only happen if the last group was for a different format than the node.
                groups.append(NodeGroup(
                    nodes=[node],
                    tensors=list(node.output),
                    format=_node_format(node)
                ))

        print(groups)
