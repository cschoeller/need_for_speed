# Introduction

The goal of this little experiment is to play with and learn about various methods to boost the training and inference speed of a simple deep neural network.

Due to hardware limitations, all evaluations were conducted on a single GPU machine with the following specifications:
- CPU: Intel Core i7-13700KF
- RAM: 32 GB DDR3
- GPU: Nvidia RTX 4090

For simplicity I chose a homebrew ConvNext for calssification of the tinyimagenet dataset.


# Experiments

This section contains the results for speeding up our model's training and inference.

## Training

We used for all training runs a batch size of 128 and images with an upsampled resolution of 224 x 224 pixels. The model itself has 18166472 parameters. The focus of these experiments is on runtimes, but we additionaly report validation accuracy as we expect this not to severely degrade with our changes.

| Changes | Train Epoch Time | Val Accuracy |
| -------- | -------- | -------- |
| Vanilla | 150.91 s | 46.53 % |
| TC | 145.69 s | 46.16 % |
| AMP | 136.97 s | 47.86 % |
| TC + AMP | 110.99 s | 47.16 % |

TC = torch.compile, AMP = Automatic Mixed Precision

## Inference

Besides picking an efficient format and inference engine, for inference evaluation it is important to be able to visualize the operations the model performs and evaluate their runtime. For this we export the model in training mode and with torch dynamo optimizations to onnx. The exported onnx files under `artifacts` can be visualized with the command `netron artifacts/<model-name>.onnx`.


# Installation Instructions

I recommend to install [conda](https://docs.conda.io/projects/miniconda/en/latest/) to run this repository. Then create and install an evironment with the these instructions:

```
conda create --name <env-name> python=3.9
conda activate <env-name>
pip install --upgrade pip
pip install -r requirements.txt
```


# Plan
- torch.compile
- mixed precision training
- torch profile for exact forward evaluation and flame chart
- Gelu Approximation
- fuse layers with trition jit

- Inference with TensorRT
- Quantization / QAT
- Print out operator chart over onnx
