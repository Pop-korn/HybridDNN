#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from itertools import chain
from typing import Callable

import numpy as np
import onnx


def _all_model_value_info(model: onnx.ModelProto) -> chain[onnx.ValueInfoProto]:
    """ Return a `chain` iterable object, which contains all ValueInfoProto objects in the given`model`. """
    return chain(model.graph.input, model.graph.output, model.graph.value_info)


def _vi_for_name(model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    """ Return the ValueInfoProto object with the given `name` from the `model`. It must be in the `model`, otherwise
         a KeyError is raised.
    """
    for vi in _all_model_value_info(model):
        if vi.name == name:
            return vi

    raise KeyError(f'Failed to find value info for `{name}`.')


def _vi_exists(model: onnx.ModelProto, name: str) -> bool:
    """ Return `True` iff a ValueInfoProto object is in the given `model` with the given `name`. """
    return len([vi for vi in _all_model_value_info(model) if vi.name == name]) > 0


def create_single_node_model(from_model: onnx.ModelProto, node: onnx.NodeProto,
                             get_tensor_data: Callable[[str], np.ndarray | None]) -> onnx.ModelProto:
    """ Create an ONNX model which is based on `from_model` but only contains 1 `node`.

    :param from_model: ONNX model containing `node` and all ValueInfoProto and Initializer objects it refers to.
    :param node: ONNX node which will be the only node in the returned model.
    :param get_tensor_data: A function which takes the name of a tensor, and returns its data or `None`. It also
                             provides access to data which is not stored in the model, but was inferred during shape
                             inference.
    :return: An ONNX model containing just the 1 given `node`.
    """

    return create_model_with_nodes(from_model, [node], node.input, node.output, get_tensor_data)


InputNames = set[str]
OutputNames = set[str]


def get_io_names_for_all_nodes(nodes: list[onnx.NodeProto]) -> (InputNames, OutputNames):
    """ Get the names of all input and output tensors for all nodes in the given list of nodes.

    :param nodes: List of nodes to analyze.
    :return: (Set of names of inputs of all nodes, Set of names of outputs of all nodes)
    """
    all_node_outputs = set()
    all_node_inputs = set()
    for node in nodes:
        all_node_inputs.update(node.input)
        all_node_outputs.update(node.output)

        if node.op_type == 'Loop':
            # The `Loop` operator has another ONNX Graph inside its `body` attribute.
            assert node.attribute[0].name == 'body'
            sub_graph = node.attribute[0].g
            # Recursively scan the subgraph, to support nested `Loop`/`If` nodes.
            loop_internal_inputs, loop_internal_outputs = get_io_names_for_all_nodes(sub_graph.node)

            # `Loop` uses inputs which are called differently in the outside graph and in its
            #  internal graph. Remove the internal inputs.
            for i in sub_graph.input:
                if i.name in loop_internal_inputs:
                    loop_internal_inputs.remove(i.name)

            all_node_inputs.update(loop_internal_inputs)
            all_node_outputs.update(loop_internal_outputs)

        elif node.op_type == 'If':
            # The `If` operator has another ONNX Graph inside its `else_branch` and `then_branch` attributes.
            assert len(node.attribute) == 2
            for attr in node.attribute:
                assert 'branch' in attr.name
                sub_graph = attr.g
                # Recursively scan the subgraph, to support nested `Loop`/`If` nodes.
                loop_internal_inputs, loop_internal_outputs = get_io_names_for_all_nodes(sub_graph.node)

                all_node_inputs.update(loop_internal_inputs)
                all_node_outputs.update(loop_internal_outputs)

    return all_node_inputs, all_node_outputs


def create_model_with_nodes(from_model: onnx.ModelProto, nodes: list[onnx.NodeProto], inputs: list[str],
                            outputs: list[str], get_tensor_data: Callable[[str], np.ndarray | None]) -> onnx.ModelProto:
    """ Create an ONNX model which is based on `from_model` and contains the given `nodes`.
         Include all the necessary `initializers`, `inputs` and `outputs`.

    :param from_model: ONNX model containing the `nodes` and all ValueInfoProto and Initializer objects they refer to.
    :param nodes: ONNX nodes which will be the only nodes in the returned model.
    :param inputs: Names of the input tensors of the created model.
    :param outputs: Names of the output tensors of the created model.
    :param get_tensor_data: A function which takes the name of a tensor, and returns its data or `None`. It also
                             provides access to data which is not stored in the model, but was inferred during shape
                             inference.
    :return: An ONNX model containing just the given `nodes`.
    """

    node_inputs, node_outputs = get_io_names_for_all_nodes(nodes)

    inputs_and_their_data = {tensor_name: get_tensor_data(tensor_name) for tensor_name in node_inputs}
    inputs_with_data = {name: data for name, data in inputs_and_their_data.items() if data is not None}
    inputs_without_data = {name for name, data in inputs_and_their_data.items() if data is None}

    # Initializers already stored in the model.
    initializers = [t for t in from_model.graph.initializer if t.name in inputs_with_data.keys()]

    if len(initializers) != len(inputs_with_data):
        # Add tensors for which we have inferred data into the initializers.
        processed_initializers = {t.name for t in initializers}
        try:
            initializers.extend([
                onnx.numpy_helper.from_array(data, name) for name, data in inputs_with_data.items() if
                name not in processed_initializers and  # Avoid adding the initializer multiple times.
                name not in node_outputs  # Node outputs cannot have static data.
            ])
            pass
        except Exception as e:
            print(e)

    graph = onnx.helper.make_graph(
        nodes,
        'model_segment',
        [_vi_for_name(from_model, input_) for input_ in inputs if input_ in inputs_without_data],
        [_vi_for_name(from_model, output) for output in outputs],
        initializers
    )

    return onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid(opset.domain, opset.version) for opset in from_model.opset_import],
    )


def _get_tensor_shape(model: onnx.ModelProto, tensor_name: str) -> list[int] | None:
    """ Return the shape of a tensor with given `tensor_name` from the `model`. """
    for vi in chain(model.graph.input, model.graph.output, model.graph.value_info):
        if vi.name == tensor_name:
            dims = vi.type.tensor_type.shape.dim
            return [dim.dim_value if hasattr(dim, 'dim_value') else '' for dim in dims]

    for t in model.graph.initializer:
        if t.name == tensor_name:
            return list(t.dims)

    return None


def _tensor_shape_is_well_defined(model: onnx.ModelProto, tensor_name: str) -> bool:
    """ Return `True` iff the shape of the tensor with given `tensor_name` contains only positive integers. """
    shape = _get_tensor_shape(model, tensor_name)
    if shape is None:
        return False

    return all(isinstance(dim, int) and dim > 0 for dim in shape)


def node_has_all_shapes_defined(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    """ Return `True` iff the shapes of all tensors used by given `node` contain only positive integers. """
    return all(_tensor_shape_is_well_defined(model, vi) for vi in chain(node.input, node.output))
