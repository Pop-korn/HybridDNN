#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    # TODO Also provide the inferred data.
    return SymbolicShapeInference.infer_shapes(model)
