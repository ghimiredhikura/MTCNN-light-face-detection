# MTCNN Light Face Detection and Evaluation

### References:

1. [MTCNN - Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
2. [MTCNN Light](https://github.com/AlphaQi/MTCNN-light)
3. [MTCNN Light Self Adaption](https://github.com/samylee/mtcnn_light_self_adaption)

This repository contains MTCNN implementation with no framework. Most of the source code is adapted from [MTCNN Light](https://github.com/AlphaQi/MTCNN-light) and [MTCNN Light Self Adaption](https://github.com/samylee/mtcnn_light_self_adaption). It is implemented in core C++ with OpenCV and openblas. Advandages of this implementations are 1. easy to use, 2. no deep learning framework need, 3. can run real time in CPU.  

### Usage

At first you need to install OpenCV2.0.x+, and openblas. 

```
git clone https://github.com/ghimiredhikura/MTCNN-light-face-detection.git
cd MTCNN-light-face-detection/
mkdir build
cd build/
cmake ..
make ./mtcnn-light
```

### Evaluation results in benchmark datasets

Dataset download: Please refer to this [git](https://github.com/bonseyes/SFD/blob/master/docs/Test-Instructions.md) for downloading dataset and evaluation source. 

#### 1. AFW, PASCAL


