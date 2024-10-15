#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from dataclasses import dataclass
from itertools import chain

import onnx
from onnx2tflite.src.converter.convert import convert_model

from model_decomposition.model_analyzer import ModelAnalyzer
from model_decomposition.onnx_model_utils import create_model_with_nodes
from model_format.hybrid_model import HybridModel, ModelFormat, ModelSegment


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

    def _create_model_segment_from_group(self, group: NodeGroup, inputs: list[str], outputs: list[str],
                                         index: int) -> ModelSegment:
        segment = ModelSegment(f'segment_{index}', group.format, inputs, outputs)

        # Create an ONNX model with the given `group.nodes` and necessary initializers.
        onnx_model = create_model_with_nodes(self.model, group.nodes, inputs, outputs)

        if group.format == ModelFormat.ONNX:
            segment.raw_data = onnx_model.SerializeToString()

        elif group.format == ModelFormat.TFLite:
            # Create an ONNX model with the given `group.nodes` and necessary initializers.
            try:
                tflite_model = convert_model(onnx_model)
                segment.raw_data = tflite_model

            except Exception as e:
                raise Exception('_create_model_segment_from_group(): failed to convert the segment to TFLite.') from e

        else:
            raise ValueError(f'_create_model_segment_from_group(): Unsupported format: {group.format}')

        return segment

    def _split_model_into_groups(self, convertible_nodes: list[onnx.NodeProto]) -> list[NodeGroup]:
        groups = [
            NodeGroup(
                nodes=[],
                tensors=[input_.name for input_ in self.model.graph.input] +  # Global model inputs.
                        [t.name for t in self.model.graph.initializer]  # Static tensors.
            )
        ]

        def _get_group_generating_tensor(tensor_name: str) -> int:
            """ Return the index of the group generating the tensor with name `tensor_name`. """
            for idx, group_ in enumerate(groups):
                if tensor_name in group_.tensors:
                    return idx

            raise KeyError

        def _get_last_group_this_node_depends_on(node_: onnx.NodeProto) -> int:
            """ Return the index of the last group of nodes that this `node_` depends on.  """
            last_group = 0
            for input_ in node_.input:
                last_group = max(_get_group_generating_tensor(input_), last_group)

            return last_group

        def _node_format(node_: onnx.NodeProto) -> ModelFormat:
            """ Get the `ModelFormat` of the given `node_`. Either `TFLite` or `ONNX`. """
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
                if group.format != _node_format(node):
                    continue

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

        return groups

    def _get_inputs_of_segment_represented_by_node_group(self, group: NodeGroup) -> set[str]:
        """ Get a set of names of tensors which are external inputs to this node group. That means all node inputs which
             are not also outputs of other nodes in this group and are not initializers.
        """
        all_node_inputs = set(chain.from_iterable(node.input for node in group.nodes))
        all_node_outputs = set(chain.from_iterable(node.output for node in group.nodes))
        initializers = set(t.name for t in self.model.graph.initializer)

        # The segment inputs are tensors which are not produced within the segment and are not static initializers.
        return all_node_inputs - all_node_outputs - initializers

    def create_hybrid_model(self):  # TODO Select decomposition strategy.
        analyzer = ModelAnalyzer(self.model)
        convertible_nodes = analyzer.get_nodes_convertible_to_tflite()
        self.model = analyzer.model

        # Each group contains nodes which will make up 1 model segment.
        node_groups = self._split_model_into_groups(convertible_nodes)

        # Determine the inputs of all individual segments, which are represented as node group.
        segment_inputs = [self._get_inputs_of_segment_represented_by_node_group(group) for group in node_groups]

        # Determine the outputs of all individual segments. That is, all tensors produced by nodes in the group, which
        #  are also consumed by other node group, or are model outputs.
        segment_outputs = []
        inputs_of_all_segments = set(chain(*segment_inputs))
        model_outputs = set(o.name for o in self.model.graph.output)
        all_node_outputs_per_group = [
            set(chain.from_iterable(node.output for node in group.nodes)) for group in node_groups
        ]
        for group_node_outputs, group in zip(all_node_outputs_per_group, node_groups):
            segment_outputs.append(
                group_node_outputs & (inputs_of_all_segments | model_outputs)
            )

        # Now for every segment, we have its nodes, inputs and outputs. So we can start constructing the model segments.
        segments = []
        for idx, (group, inputs, outputs) in enumerate(zip(node_groups, segment_inputs, segment_outputs)):
            segments.append(self._create_model_segment_from_group(group, list(inputs), list(outputs), idx))

        # Combine the segments into a `HybridModel`.
        hybrid_model = HybridModel()
        hybrid_model.model_segments = segments

        return hybrid_model
