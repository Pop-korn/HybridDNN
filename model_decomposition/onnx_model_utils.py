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

    raise KeyError


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

    node_inputs = set(chain.from_iterable(node.input for node in nodes))

    inputs_and_their_data = {tensor_name: get_tensor_data(tensor_name) for tensor_name in node_inputs}
    inputs_with_data = {name: data for name, data in inputs_and_their_data.items() if data is not None}
    inputs_without_data = {name for name, data in inputs_and_their_data.items() if data is None}
    if not all(_vi_exists(from_model, name) for name in inputs_without_data):
        raise Exception('create_model_with_nodes(): the source model does not contain the necessary value info.')

    # Initializers already stored in the model.
    initializers = [t for t in from_model.graph.initializer if t.name in inputs_with_data.keys()]

    if len(initializers) != len(inputs_with_data):
        # Add tensors for which we have inferred data into the initializers.
        processed_initializers = {t.name for t in initializers}
        initializers.extend([
            onnx.numpy_helper.from_array(data, name) for name, data in inputs_with_data.items() if
            name not in processed_initializers
        ])

    graph = onnx.helper.make_graph(
        nodes,
        'model_segment',
        [_vi_for_name(from_model, input_) for input_ in inputs if input_ in inputs_without_data],
        [_vi_for_name(from_model, output) for output in outputs],
        initializers
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
