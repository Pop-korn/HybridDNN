# HybridDNN - Framework for acceleration of ONNX models via partial conversion to LiteRT

This framework enables partitioning of ONNX deep neural network models into multiple segments. Some of these segments are quantized and converted to the LiteRT format to leverage hardware accelerators on mobile and embedded platforms by using existing runtime infrastructure.

## Features

- Partition ONNX models into ordered disjoint segments.
- Quantize selected segments.
- Convert some segments to the LiteRT format for acceleration.
- Recombine segments into a single *hybrid model* for seamless inference.
- Execute and profile hybrid models using hardware acceleration where available.

## Directory Structure

- `src/model_decomposition/` – Tools for partitioning ONNX models and creating *hybrid models*.
- `src/model_format/` –  Loading, storing, and in-memory representation of *hybrid models*.
- `src/model_inference/` – Inference engine and profiling utilities for *hybrid models*.

## Getting Started

### Prerequisites

Install and set up the [NXP eIQ Toolkit](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT) to access its custom Python interpreter. On Ubuntu, the interpreter is installed into `/opt/nxp/eIQ_Toolkit_v<version number>/python/bin/python`. This is the interpreter that must be used with this framework, as it provides access to the required packages `onnx2tflite` and `onnx2quant`.

Then install dependencies via:

```bash
pip install -r requirements.txt
```

### Example usage

An example SSD ONNX model is provided in the `data/ssd` directory. Two example scripts were created to demonstrate the functionality provided by the *HybridDNN* framework.

* The model can be decomposed into a *hybrid model* using the script `create_hybrid_model.py`. The script provides an overview on how the framework can be used to create *hybrid models* from any input ONNX models. When running the script, it is normal to see a large number of error and warning messages. These are caused by the expected failures of third party tools used during the model decomposition, and they cannot be suppressed.
    ```bash
    python create_hybrid_model.py
    ```

* The script `run_hybrd_model.py` runs inference of the aforementioned model, and prints profiling statistics. The purpose of the script is to provide an overview of how the framework can be used to run any *hybrid models*.
    ```bash
    python run_hybrid_model.py
    ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.