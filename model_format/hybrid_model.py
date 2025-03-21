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
    LiteRT = 'litert'


class ModelSegment:
    """ Class represents a part of the original model. Each segment is either in the ONNX or LiteRT format. """

    file_name: str
    format: ModelFormat
    inputs: list[str]
    outputs: list[str]
    raw_data: bytes  # The binary model.

    def __init__(self, file_name: str | None = None, format_: ModelFormat | None = None,
                 inputs: list[str] | None = None, outputs: list[str] | None = None,
                 raw_data: bytes | None = None) -> None:
        if file_name is not None and format_ is not None:
            file_extension = '.tflite' if format_ == ModelFormat.LiteRT else '.onnx'
            self.file_name = file_name + file_extension
        self.format = format_
        self.inputs = inputs
        self.outputs = outputs
        self.raw_data = raw_data

    @classmethod
    def from_json(cls, json_segment: dict[str, str | list[str]], data: bytes) -> 'ModelSegment':
        """ Create a `ModelSegment` from the provided raw `data` and the metadata in JSON format.

        :param json_segment: JSON metadata of the segment.
        :param data: Raw data of the DNN model in the segment.
        :return: A new `ModelSegment` instance.
        """
        try:
            obj = cls()

            obj.file_name = json_segment['name']
            obj.format = obj._parse_model_format(obj.file_name)
            obj.inputs = json_segment['inputs']
            obj.outputs = json_segment['outputs']
            obj.raw_data = data

            return obj

        except KeyError as e:
            if (file_name := json_segment.get('name', None)) is None:
                message = 'Invalid format of `meta.json` file. Failed to parse the name of a model segment.'
            else:
                message = f'Invalid format of `meta.json` file. Failed to parse model segment `{file_name}`.'

            raise KeyError(message) from e

    def _parse_model_format(self, name: str) -> ModelFormat:
        """ Get a valid `ModelFormat` value from a string representation of the formats associated file extension. """
        extension = os.path.splitext(self.file_name)[1][1:]
        match extension:
            case 'tflite':
                return ModelFormat.LiteRT
            case 'onnx':
                return ModelFormat.ONNX
            case _:
                raise ValueError(f'Model segment `{name}` has an unsupported file format.')

    def to_json(self) -> dict[str, str | list[str]]:
        """ Return a JSON representation of this ModelSegment.

        :return: JSON representation of this model segments, represented as dicts and lists of strings.
        """
        return {
            'name': self.file_name,
            'inputs': self.inputs,
            'outputs': self.outputs
        }

    def pretty_segment_size(self) -> str:
        """ Return the size of the model segment as a string with "pretty" formatting. """
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = len(self.raw_data)
        for unit in units:
            if size < 1024:
                return f'{size:0.2f} {unit}'
            size /= 1024

        # Ran out of units (should never happen).
        return f'{len(self.raw_data)} {units[0]}'

    def __repr__(self) -> str:
        """ Represent this segment in a debug string. """
        return f'''\t- `{self.file_name}` ({self.format.name}) - {self.pretty_segment_size()}
        Inputs: {self.inputs}
        Outputs: {self.outputs}
'''


class HybridModel:
    """ Class represents a DNN model which is divided into multiple segments in different formats. These segments can
         be connected together to reconstruct the original model.
    """

    model_segments: list[ModelSegment]
    inputs: list[str]
    outputs: list[str]

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

                # Parse the inputs and outputs of the whole graph.
                self.inputs = meta['inputs']
                self.outputs = meta['outputs']

                # Parse the individual model segments.
                for segment in meta['segments']:
                    with zf.open(segment['name'], 'r') as segment_file:
                        self.model_segments.append(ModelSegment.from_json(segment, segment_file.read()))

        except KeyError as e:
            raise KeyError(f'Failed to parse the hdnn file `{path}`. It is not in the expected format.') from e
        except zipfile.BadZipFile as e:
            raise Exception(
                f"The provided file `{path}` couldn't be parsed, because it is not in the `.hdnn` format.") from e

    def store(self, path: str):
        """ Store the data in `self.model_segments` in a `zip` file. """

        # First create the JSON `meta file` based on the data in the individual segments.
        meta = {
            'inputs': self.inputs,
            'outputs': self.outputs,
            'segments': [
                segment.to_json() for segment in self.model_segments
            ]
        }

        # Save the file in the zip.
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save the `meta`.
            zf.writestr('meta.json', json.dumps(meta))

            # Save the individual LiteRT and ONNX model segments.
            for segment in self.model_segments:
                zf.writestr(segment.file_name, segment.raw_data)

    def __repr__(self) -> str:
        """ Represent the HybridModel in a debug string. """
        description = f'Model of {len(self.model_segments)} segments.\n'
        segments_description = [repr(segment) for segment in self.model_segments]

        return description + ''.join(segments_description)

    def save_segments(self, directory: str) -> None:
        """ Store all the internal model segments in the provided directory. This is used only for debugging purposes.

        :param directory: Directory to store the segments in.
        """
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        for segment in self.model_segments:
            file_path = os.path.join(directory, segment.file_name)
            with open(file_path, 'wb') as f:
                f.write(segment.raw_data)
