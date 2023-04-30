# ONNX runtime for node with gpu support (DirectML/Cuda)

## Info
This is an updated copy of official onnxruntime-node with DirectML and Cuda support.

## Requirements
### Windows
1. Works out of the box with DirectML. You can install CUDA and onnx runtime for windows with cuda provider for experiments, if you like.
### Linux / WSL2
1. Install CUDA (tested only on 11-7 but 12 should be supported) https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
2. Install onnxruntime-linux-x64-gpu-1.14.1 https://github.com/microsoft/onnxruntime/releases/tag/v1.14.1

## Installation and usage
It works in the same way as [onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node) 
```
npm i onnxruntime-node-gpu
```
```
import { InferenceSession, Tensor } from 'onnxruntime-node-gpu'

const sessionOption: InferenceSession.SessionOptions = { executionProviders: ['directml'] } // can be also 'cuda' or 'cpu'
const model = await InferenceSession.create('model.onnx', sessionOption)

const input = new Tensor('float32', Float32Array.from([0, 1, 2]), [3])
const result = await this.textEncoder.run({ input_name: input })
```

## Limitations
1. Currently, all results are returned as NAPI nodejs objects, so when you run inference multiple times (e.g. sampling on StableDiffusion Unet),
there are a lot of unnecessary memory copy operations input from js to gpu and back. However, performance impact is not big. 
Maybe later I will make output in Tensorflow.js compatible tensors

## Building manually
Just download the repo and run `npx cmake-js compile`

## Why is onnxruntime statically linked on Windows?
For some reason, dynamically linked onnx runtime tries to load outdated DirectML.dll in system32, see https://github.com/royshil/obs-backgroundremoval/issues/272

## Misc
Special thanks to authors of https://github.com/royshil/obs-backgroundremoval and https://github.com/umireon/onnxruntime-static-win
for CMake scripts to download pre-built onnxruntime for static linking.

Also thanks to ChatGPT for helping me to remember how to code in c++.

You can ask me questions on [Twitter](https://twitter.com/daken_)