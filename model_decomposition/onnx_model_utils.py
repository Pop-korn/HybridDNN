#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from itertools import chain

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

    raise KeyError


def _vi_exists(model: onnx.ModelProto, name: str) -> bool:
    """ Return `True` iff a ValueInfoProto object is in the given `model` with the given `name`. """
    return len([vi for vi in _all_model_value_info(model) if vi.name == name]) > 0


def create_single_node_model(from_model: onnx.ModelProto, node: onnx.NodeProto) -> onnx.ModelProto:
    """ Create an ONNX model which is based on `from_model` but only contains 1 `node`.

    :param from_model: ONNX model containing `node` and all ValueInfoProto and Initializer objects it refers to.
    :param node: ONNX node which will be the only node in the returned model.
    :return: An ONNX model containing just the 1 given `node`.
    """
    graph = onnx.helper.make_graph(
        [node],
        'single_node_model',
        [_vi_for_name(from_model, input_) for input_ in node.input if _vi_exists(from_model, input_)],
        [_vi_for_name(from_model, output) for output in node.output if _vi_exists(from_model, output)],
        [t for t in from_model.graph.initializer if t.name in node.input],
    )

    return onnx.helper.make_model(graph)


def create_model_with_nodes(from_model: onnx.ModelProto, nodes: list[onnx.NodeProto], inputs: list[str],
                            outputs: list[str]) -> onnx.ModelProto:
    """ Create an ONNX model which is based on `from_model` and contains the given `nodes`.
         Include all the necessary `initializers`, `inputs` and `outputs`.

    :param from_model: ONNX model containing the `nodes` and all ValueInfoProto and Initializer objects they refer to.
    :param nodes: ONNX nodes which will be the only nodes in the returned model.
    :param inputs: Names of the input tensors of the created model.
    :param outputs: Names of the output tensors of the created model.
    :return: An ONNX model containing just the given `nodes`.
    """
    if not all(_vi_exists(from_model, name) for name in chain(inputs, outputs)):
        raise Exception('create_model_with_nodes(): the source model does not contain the necessary value info.')

    node_inputs = set(chain.from_iterable(node.input for node in nodes))
    used_initializers = [t for t in from_model.graph.initializer if t.name in node_inputs]

    graph = onnx.helper.make_graph(
        nodes,
        'model_segment',
        [_vi_for_name(from_model, input_) for input_ in inputs],
        [_vi_for_name(from_model, output) for output in outputs],
        used_initializers
    )

    return onnx.helper.make_model(graph)


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
