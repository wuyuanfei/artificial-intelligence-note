# 计算机视觉

[TOC]

#### 1. 书籍推荐

- 《Computer Vision：Models, Learning and Inference》
- 《Computer Vision：Algorithms and Applications》
- 《OpenCV3编程入门》
- 《DEEP LEARNING》

#### 2. 在线课程

- Stanford CS223B
- Stanford CS231N

#### 3. 开源软件

- OpenCV
- Caffe
- TensorFlow

#### 4. 计算机视觉顶会

- ICCV：International Conference on Computer Vision，国际计算机视觉大会
- CVPR：International Conference on Computer Vision and Pattern Recognition，国际计算机视觉与模式识别大会
- ECCV：European Conference on Computer Vision，欧洲计算机视觉大会

#### 5. 项目或竞赛

- **物体分类（Object Classification）**
  - CNN
- **目标检测（Object Detection）**
  - RCNN、Fast RCNN、Faster RCNN
  - SPPNET
  - SSD
  - YOLOv1、YOLOv2、YOLOv3
- **目标跟踪**
  - DLT、SO-DLT
- **人脸检测（Face recognition）**
  - FaceNet、MTCNN
- **语义分割（Image Semantic Segmentation）**
  - FCN

#### 6. 优秀github项目

[人工智能工程师面试](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese)
[dingdang智能音箱](https://github.com/dingdang-robot/dingdang-robot)    

#### 7. 工作项目

- **Text detection**

  - **CPTN**：https://github.com/eragonruan/text-detection-ctpn
  - http://slade-ruan.me/
  - **CNN+RNN+CTC**方案
  - https://zhuanlan.zhihu.com/p/29549641
  - **ICDAR2015 - Incidental Scene Text dataset**
    - 常用的英文文字检测数据集

    - 它涵盖1000张训练图片（约包含4500个单词）和500张测试图片；
    - 它重点采集了一些随机场景，在这些场景中文字具有方向任意、字体小、低像素的特性。
  - **MSRA-TD500**
    - 中英数据集
    - 包含了500张自然图片（涵盖室内、室外采集）；
    - 包含中文、英文及中英混合形式，具有不同的字体、大小、颜色、方向；
    - 文本边框标注；
  - **RCTW-17**
    - 包含中文文本的图片共12034张（其中8034张训练图片，4000张测试图片）；
    - 图片涵盖汉字、数字、英文单词，其中汉字占最大比例；
    - ICDAR2017的中文场景文字检测比赛用的是这个数据集。
  - 

- **Face recognition**

  - opencv Haar

    > 有点是简单，快速；存在的问题是人脸检测效果不好。如图3-1所示，正面/垂直/光线较好的人脸，该方法可以检测出来，而侧面/歪斜/光线不好的人脸，无法检测。因此，该方法不适合现场应用

  - dlib

    > 效果好于opencv的方法，但是检测力度也难以达到现场应用标准

  - mtcnn + facenet

    > mtcnn人脸检测方法对自然环境中光线，角度和人脸表情变化更具有鲁棒性，人脸检测效果更好；同时，内存消耗不大，可以实现实时人脸检测

  - 训练集：VGGFace2数据集

    > 331万个图像=9131实例*362.6图像
    >
    > 姿势，年龄，种族，职业（演员，运动员，政治家）
- **object detection**
  - ssd
  - yolov3
  - moblenet
