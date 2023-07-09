# DepthEstimationEndoscopy
Dense Depth Estimation in Monocular Endoscopy with Self-supervised Learning Methods

## Requirements
* This code is tested with Keras 2.2.4, Tensorflow 1.13, CUDA 10.0, on a machine with an NVIDIA Titan V and 16GB+ RAM running on Windows 10 or Ubuntu 16.
* Other packages needed `keras pillow matplotlib scikit-learn scikit-image opencv-python pydot` and `GraphViz` for the model graph visualization and `PyGLM PySide2 pyopengl` for the GUI demo.
* Minimum hardware tested on for inference NVIDIA GeForce 940MX (laptop) / NVIDIA GeForce GTX 950 (desktop).
* Training takes about 24 hours on a single NVIDIA TITAN RTX with batch size 8.

## Testing
* Run `python test.py`
* This will test on the images in the examples directory or other files specifically specified in the test.py code.

## Training
* Run `python train.py --data nyu --gpus 4 --bs 8`.
