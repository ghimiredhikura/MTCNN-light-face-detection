# MTCNN Light Face Detection and Evaluation

This repository contains MTCNN implementation with no framework. Most of the source code is adapted from [MTCNN Light](https://github.com/AlphaQi/MTCNN-light) and [MTCNN Light Self Adaption](https://github.com/samylee/mtcnn_light_self_adaption). It is implemented in core C++ with OpenCV and openblas. Advandages of this implementations are 1. easy to use, 2. no deep learning framework need, 3. real time runtime performance in CPU.  

### 1. Usage

At first you need to install OpenCV2.0.x+, and openblas. 

```
$git clone https://github.com/ghimiredhikura/MTCNN-light-face-detection.git
$cd MTCNN-light-face-detection/
$mkdir build
$cd build/
$cmake ..
$make 
```
#### 1. Test webcam
```
$./mtcnn-light -mode=0 -webcam=0
```
#### 2. Test single image
```
$./mtcnn-light -mode=1 -path=../image/1.jpg
```
#### 3. Test image lists
```
$./mtcnn-light -mode=2 -path=../image/
```
#### 4. Evaluation in benchmark dataset, detection files will be stored in "detections" folder. 
```
a) afw dataset
$./mtcnn-light -mode=3 -dataset=AFW -path=/path/to/afw/dataset/

b) PASCAL dataset
$./mtcnn-light -mode=3 -dataset=PASCAL -path=/path/to/pascal/dataset/

c) FDDB dataset
$./mtcnn-light -mode=3 -dataset=FDDB -path=/path/to/fddb/dataset/

d) WIDER_val dataset
#./mtcnn-light -mode=3 -dataset=WIDER_VAL -path=/path/to/wider/validation/dataset/

e) UFDD dataset
#./mtcnn-light -mode=3 -dataset=UFDD -path=/path/to/UFDD/validation/dataset/
```

### 2. Evaluation results in benchmark datasets

Dataset download: Please refer to this [git](https://github.com/bonseyes/SFD/blob/master/docs/Test-Instructions.md) for downloading dataset and evaluation tools. 

#### a. [AFW](http://www.ics.uci.edu/~xzhu/face/), PASCAL ([train-val](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), [test](http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/)) and [FDDB](http://vis-www.cs.umass.edu/fddb/index.html)
![Alt text](image/mtcnn-sfd_afw_pascal_fddb.PNG)
#### b. [WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
![Alt text](https://github.com/bonseyes/SFD/blob/master/docs/assets/WIDER_sfd-mtcnn.PNG)

### 3. References:

1. [MTCNN - Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
2. [MTCNN Light](https://github.com/AlphaQi/MTCNN-light)
3. [MTCNN Light Self Adaption](https://github.com/samylee/mtcnn_light_self_adaption)
4. [S³FD: Single Shot Scale-invariant Face Detector](https://github.com/bonseyes/SFD)
