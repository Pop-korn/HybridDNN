#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from dataclasses import dataclass
from enum import Enum
from itertools import chain

import numpy as np
import onnx
import sympy
from onnx2tflite.src.converter.convert import convert_model

from model_decomposition.model_analyzer import ModelAnalyzer
from model_decomposition.onnx_model_utils import create_model_with_nodes
from model_format.hybrid_model import HybridModel, ModelFormat, ModelSegment


class DecompositionStrategy(Enum):
    NAIVE = 1  # Convert every operator that can be converted.

    # Convert all nodes which can utilize HW accelerators to `LiteRT`, while minimizing the number of model segments.
    RATIONAL = 2


@dataclass
class NodeGroup:
    nodes: list[onnx.NodeProto]
    tensors: list[str]  # List of names of tensors which are produced by this group of nodes.
    format: ModelFormat | None = None


class ModelDecomposer:
    """ Class divides the input ONNX model into sections. Some of them get converted to LiteRT and some stay in ONNX.
         These sections are then combined into a hybrid model (hdnn).
    """

    model: onnx.ModelProto

    def __init__(self, model: onnx.ModelProto | str):
        if isinstance(model, str):
            self.model = onnx.load(model)
        else:
            self.model = model

        self.analyzer = ModelAnalyzer(self.model)

    def _get_tensor_data(self, tensor_name: str) -> np.ndarray | None:
        """ Get the static or inferred data of a given tensor, if that data is available. """

        def _get_nested_initializers():
            if not hasattr(_get_nested_initializers, 'initializers'):
                _get_nested_initializers.initializers = []
                for node in self.model.graph.node:
                    if node.op_type == 'Loop':
                        # The `Loop` operator contains an entire subgraph as its `body` attribute. Add its initializers
                        #  to the list.
                        assert node.attribute[0].name == 'body', 'Unexpected `Loop` attribute.'
                        sub_graph = node.attribute[0].g
                        _get_nested_initializers.initializers.extend(sub_graph.initializer)

            return _get_nested_initializers.initializers

        sympy_data = self.analyzer.shape_inference.get_tensor_data(tensor_name)
        if isinstance(sympy_data, sympy.Symbol):  # The `Shape` operator can produce a `Symbol`.
            sympy_data = None

        if sympy_data is None:
            nested_initializer = [i for i in _get_nested_initializers() if i.name == tensor_name]
            if len(nested_initializer) == 1:
                return nested_initializer[0]

        return sympy_data

    def _create_model_segment_from_group(self, group: NodeGroup, inputs: list[str], outputs: list[str],
                                         index: int) -> ModelSegment:
        segment = ModelSegment(f'segment_{index}', group.format, inputs, outputs)

        # Create an ONNX model with the given `group.nodes` and necessary initializers.
        onnx_model = create_model_with_nodes(self.model, group.nodes, inputs, outputs, self._get_tensor_data)

        if group.format == ModelFormat.ONNX:
            segment.raw_data = onnx_model.SerializeToString()

        elif group.format == ModelFormat.LiteRT:
            # Create an ONNX model with the given `group.nodes` and necessary initializers.
            try:
                litert_model = convert_model(onnx_model)
                segment.raw_data = litert_model

            except Exception as e:
                raise Exception('_create_model_segment_from_group(): failed to convert the segment to LiteRT.') from e

        else:
            raise ValueError(f'_create_model_segment_from_group(): Unsupported format: {group.format}')

        return segment

    def _split_model_into_groups(self, nodes_to_convert: list[onnx.NodeProto]) -> list[NodeGroup]:
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
            """ Get the `ModelFormat` of the given `node_`. Either `LiteRT` or `ONNX`. """
            return ModelFormat.LiteRT if node_ in nodes_to_convert else ModelFormat.ONNX

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
                break

            if not node_added:
                # There is no existing group suitable for `node`. Create a new group and add the `node` to it.
                # This can only happen if the last group was for a different format than the node.
                groups.append(NodeGroup(
                    nodes=[node],
                    tensors=list(node.output),
                    format=_node_format(node)
                ))

        return groups

    def _get_external_inputs_of_node_group(self, group: NodeGroup) -> set[str]:
        """ Get a set of names of tensors which are external inputs to this node group. That means all node inputs which
             are not also outputs of other nodes in this group and are not initializers.
        """
        all_node_outputs = set()  # set(chain.from_iterable(node.output for node in group.nodes))
        all_node_inputs = set()  # set(chain.from_iterable(node.input for node in group.nodes))
        for node in group.nodes:
            all_node_inputs.update(node.input)
            all_node_outputs.update(node.output)

            if node.op_type == 'Loop':
                # The `Loop` operator has another ONNX Graph inside its `body` attribute.
                assert node.attribute[0].name == 'body'
                sub_graph = node.attribute[0].g
                loop_internal_inputs = set()
                loop_internal_outputs = set()
                for node_ in sub_graph.node:
                    if node_.op_type == 'Loop':
                        raise NotImplementedError('Nested `Loop` operators are not supported.')
                    loop_internal_inputs.update(node_.input)
                    loop_internal_outputs.update(node_.output)

                # `Loop` uses inputs which are called differently in the outside graph and in its
                #  internal graph. Remove the internal inputs, as they are duplicates.
                for i in sub_graph.input:
                    if i.name in loop_internal_inputs:
                        loop_internal_inputs.remove(i.name)

                all_node_inputs.update(loop_internal_inputs)
                all_node_outputs.update(loop_internal_outputs)

        # The segment inputs are tensors which are not produced within the segment and do NOT have static data.
        return set(t for t in all_node_inputs - all_node_outputs if self._get_tensor_data(t) is None)

    def _create_model_segments_from_node_groups(self, node_groups: list[NodeGroup]) -> list[ModelSegment]:
        # Determine the inputs of all individual segments, which are represented as node groups.
        segment_inputs = [self._get_external_inputs_of_node_group(group) for group in node_groups]

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

        return segments

    def _merge_litert_node_groups_without_accelerable_nodes_with_neighboring_groups(self, node_groups: list[NodeGroup],
                                                                                    ) -> list[NodeGroup]:
        nodes_to_convert = []
        for node_group in node_groups:
            if node_group.format != ModelFormat.LiteRT:
                continue

            if any(self.analyzer.node_will_be_accelerated_in_litert(node) for node in node_group.nodes):
                # Some nodes can be accelerated in this group, so it should still get converted to `LiteRT`.
                nodes_to_convert.extend(node_group.nodes)

            else:
                # This group should be left in `ONNX`.
                pass

        return self._split_model_into_groups(nodes_to_convert)

    def create_hybrid_model(self, decomposition_strategy: DecompositionStrategy = DecompositionStrategy.NAIVE
                            ) -> HybridModel:
        convertible_nodes = self.analyzer.get_nodes_convertible_to_litert()
        self.model = self.analyzer.model

        # Divide the model into groups of nodes.
        node_groups = self._split_model_into_groups(convertible_nodes)

        if decomposition_strategy == DecompositionStrategy.RATIONAL:
            # Search for `LiteRT` groups which only contain nodes that don't use HW accelerators. Remove these groups
            #  and merge their operators into neighbouring `ONNX` groups.
            node_groups = self._merge_litert_node_groups_without_accelerable_nodes_with_neighboring_groups(node_groups)

        # Each group contains nodes which will make up 1 model segment.
        segments = self._create_model_segments_from_node_groups(node_groups)

        # Combine the segments into a `HybridModel`.
        hybrid_model = HybridModel()
        hybrid_model.inputs = [vi.name for vi in self.model.graph.input]
        hybrid_model.outputs = [vi.name for vi in self.model.graph.output]
        hybrid_model.model_segments = segments

        return hybrid_model
