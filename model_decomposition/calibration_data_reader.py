#
# Copyright 2025 Martin Pavella
#
# License: MIT
# See the LICENSE for more details.
#

from typing import Iterator

import datasets
import numpy as np
from PIL import Image
from onnx2quant.qdq_quantization import CalibrationDataReader


class DatasetReader(CalibrationDataReader):
    """ Simple class for accessing datasets using the `datasets` library. """
    iterable_data: Iterator[dict[str, np.ndarray]]

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """ Preprocess a single instance of input data. Child classes can overwrite this method.

            :param data: Data to preprocess.
            :return: Preprocessed data.
        """
        return data

    def _load_image(self, path: str, size: tuple[int, int] = (1200, 1200),
                    resample: int | None = Image.BILINEAR) -> np.ndarray:
        """ Load an image from the specified path, preprocess is, and reutrn as an array.

        :param path: Path to the file to load.
        :param size: (height, width) that the image will be resized to.
        :param resample: Interpolation type from the `PIL.Image` module.
        :return: The image as a preprocessed numpy array.
        """
        img = Image.open(path)
        img = img.resize(size, resample)
        img_data = np.array(img)
        return self._preprocess(img_data)

    def __init__(self, path: str, name: str, data_dir: str) -> None:
        super().__init__()
        ds = datasets.load_dataset(path, name, data_dir=data_dir)
        self.iterable_data = iter({'image': self._load_image(image['image_path'])} for image in ds['test'])

    def get_next(self) -> dict[str, np.ndarray]:
        """ Get the next set of input data. """
        return next(self.iterable_data, None)


class COCOReader(DatasetReader):

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        # Override parent method.
        data = np.transpose(data, [2, 0, 1])
        data = np.expand_dims(data, 0)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_data = np.zeros(data.shape).astype('float32')
        for i in range(data.shape[1]):
            norm_data[:, i, :, :] = (data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        return norm_data

    def __init__(self) -> None:
        super().__init__("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
