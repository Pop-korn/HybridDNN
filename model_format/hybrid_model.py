#
# Copyright 2024 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

import json
import os.path
import zipfile

from enum import Enum


class ModelFormat(Enum):
    ONNX = 'onnx'
    TFLite = 'tflite'


class ModelSegment:
    """ Class represents a part of the original model. Each segment is either in the ONNX or TFLite format. """

    file_name: str
    format: ModelFormat
    inputs: list[str]
    outputs: list[str]
    raw_data: bytes  # The binary model.

    def __init__(self, json_segment: dict[str, str | list[str]], data: bytes) -> None:
        try:
            self.file_name = json_segment['name']
            self.format = self._parse_model_format(self.file_name)
            self.inputs = json_segment['inputs']
            self.outputs = json_segment['outputs']
            self.raw_data = data


        except KeyError as e:
            if not hasattr(self, 'name'):
                message = 'Invalid format of `meta.json` file. Failed to parse the name of a model segment.'
            else:
                message = f'Invalid format of `meta.json` file. Failed to parse model segment `{self.file_name}`.'

            raise KeyError(message) from e

    def _parse_model_format(self, name: str) -> ModelFormat:
        try:
            extension = os.path.splitext(self.file_name)[1][1:]
            return ModelFormat(extension)

        except ValueError:
            raise ValueError(f'Model segment `{name}` has an unsupported file format.')

    def to_json(self) -> dict[str, str | list[str]]:
        return {
            'name': self.file_name,
            'inputs': self.inputs,
            'outputs': self.outputs
        }


class HybridModel:
    """ Class represents a DNN model which is divided into multiple segments in different formats. These segments can
         be connected together to reconstruct the original model.
    """

    model_segments: list[ModelSegment]

    def __init__(self, path: str | None = None):
        """ If `path` is provided, the model is also immediately loaded. """
        self.model_segments = []
        if path is not None:
            self.load(path)

    def load(self, path: str):
        """ Load and parse the `zip` file into the individual model segments. """
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                # Parse the `meta file`.
                with zf.open('meta.json', 'r') as meta_file:
                    json_data = meta_file.read().decode('utf-8')
                    meta = json.loads(json_data)

                # Parse the individual model segments.
                for segment in meta['segments']:
                    with zf.open(segment['name'], 'r') as segment_file:
                        self.model_segments.append(ModelSegment(segment, segment_file.read()))

        except KeyError as e:
            raise KeyError(f'Failed to parse the hdnn file `{path}`. It is not in the expected format.') from e

    def store(self, path: str):
        """ Store the data in `self.model_segments` in a `zip` file. """

        # First create the JSON `meta file` based on the data in the individual segments.
        meta = {
            'segments': [
                segment.to_json() for segment in self.model_segments
            ]
        }

        # Save the file in the zip.
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save the `meta`.
            zf.writestr('meta.json', json.dumps(meta))

            # Save the individual TFLite and ONNX model segments.
            for segment in self.model_segments:
                zf.writestr(segment.file_name, segment.raw_data)
