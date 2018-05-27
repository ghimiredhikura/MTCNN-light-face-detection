# MTCNN Light Face Detection and Evaluation

This repository contains MTCNN implementation with no framework. Most of the source code is adapted from [MTCNN Light](https://github.com/AlphaQi/MTCNN-light) and [MTCNN Light Self Adaption](https://github.com/samylee/mtcnn_light_self_adaption). It is implemented in core C++ with OpenCV and openblas. Advandages of this implementations are 1. easy to use, 2. no deep learning framework need, 3. real time runtime performance in CPU.  

### 1. Usage

At first you need to install OpenCV2.0.x+, and openblas. 

```
git clone https://github.com/ghimiredhikura/MTCNN-light-face-detection.git
cd MTCNN-light-face-detection/
mkdir build
cd build/
cmake ..
make ./mtcnn-light
```

### 2. Evaluation results in benchmark datasets

Dataset download: Please refer to this [git](https://github.com/bonseyes/SFD/blob/master/docs/Test-Instructions.md) for downloading dataset and evaluation tools. 

#### a. [AFW](http://www.ics.uci.edu/~xzhu/face/), [PASCAL-TRAIN/VAL](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), [PASCAL-TEST](http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/) and FDDB(http://vis-www.cs.umass.edu/fddb/index.html)

#### b. [WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

### 3. References:

1. [MTCNN - Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
2. [MTCNN Light](https://github.com/AlphaQi/MTCNN-light)
3. [MTCNN Light Self Adaption](https://github.com/samylee/mtcnn_light_self_adaption)
4. [S³FD: Single Shot Scale-invariant Face Detector](https://github.com/bonseyes/SFD)
