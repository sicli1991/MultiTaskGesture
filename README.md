# Simultaneous prediction of hand gestures, handedness, and hand keypoints using thermal images

## Abstract
Hand gesture detection is a well-explored area in computer vision with applications in various forms of Human-Computer Interactions. In this work, we propose a technique for simultaneous hand gesture classification, handedness detection, and hand keypoints localization using thermal data captured by an infrared camera. Our method uses a novel deep multi-task learning architecture that includes shared encoder-decoder layers followed by three branches dedicated for each mentioned task. We performed extensive experimental validation of our model on an in-house dataset consisting of 24 users' data. The results confirm higher than 98% accuracy for gesture classification, handedness detection, and fingertips localization, and more than 91% accuracy for wrist points localization.

<p align='center'>
  <img width='300' height='200' src="example.gif"/>
</p>
 

## ***Dataset***
Data captured with Viento-G thermal camera. We use background subtraction to detect binary hand regions, k-means clustering to isolate each hand, crop the image to tightly include each hand region, and resize it to a 100x100 pixel image.

<p align='center'>
<img width="600" alt="image" src="https://github.com/sicli1991/MultiTaskGesture/assets/55030732/f5480ed5-6fe0-4cee-9e77-6fa5ebc01fb3">
</p>

### ***Raw Data***

<img hspace="15" width="300" alt="image" src="https://github.com/sicli1991/MultiTaskGesture/assets/55030732/b0e518d2-5636-4ade-a6ee-b1bfb68e7f75">

* We have 24 users(with both left and right hand)
* Video captured in 30fps
* Saved as 16 bit  640×480 TIFF  

> ***Download from Google Drive*** [HERE](https://drive.google.com/file/d/1DaoVD-vdYuS9y7XGbFRgaxQd4y2tbIgH/view?usp=share_link)

### ***Cropped Data***

<img width="666" alt="image" src="https://user-images.githubusercontent.com/55030732/235724880-1ae363f1-e97c-4d56-93c1-4d8fc1404593.png">
&ensp;DATA:  100x100 Binary TIFF Image<br/>

&ensp;LABEL: [[(WristPoint1), (WristPoint2), (ForearmPoint)][(ThumbFingertip), (IndexFingertip), (MiddleFingertip), </br> 
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;(RingFingertip), (LittleFingertip)]]<br/>

>**Concealed Point marked as (9999,9999)**<br/> 

> ***Download from Google Drive*** [HERE](https://drive.google.com/file/d/1XCc8_XF3VJBpRaXtiawVCsl2Ot1vUe4J/view?usp=share_link)<br/>

## ***Data Label***

<p align='center'>
  <img width=60% height=60% src="https://github.com/sicli1991/MultiTaskGesture/assets/55030732/7fefbe78-4528-47a0-88a0-dc7d3224d6ee">
  <img width=60% height=60% src="https://github.com/sicli1991/MultiTaskGesture/assets/55030732/f0429062-1ae7-474d-a1b9-29b4f266aea6">
</p>

## ***Dataset Distribution***

<p align='center'>
  <img width=600 height=150 src="https://github.com/sicli1991/MultiTaskGesture/assets/55030732/0968ee3a-5d3c-4eb1-aa65-3133d787033d"/>
 </p>
 
 <p align='center'>
  <img height="150" src="https://github.com/sicli1991/MultiTaskGesture/assets/55030732/03cfd0ff-8713-42cd-8141-487906febcbb"/>
 </p>

## ***Model Structure***
<p align='center'>
  <img height="350" src="https://github.com/sicli1991/MultiTaskGesture/assets/55030732/9c66df1f-eb83-463f-b977-940aeef34157"/>
</p>
The model expects a fixed size input of size 100 × 100. The shared part of the model is similar to a U-Net architecture, and it consists of four convolutional layers, four max-pooling layers, and two up-convolutional layers. The encoder part of the network uses a series of max-pooling and convolutional layers to down-sample, the resolution to 6 × 6. This is followed by two up-convolution layers with skip connections to increase the resolution back to 25 × 25. <br/>
The top two branches of the network perform the gesture classification and left-right hand detection tasks. They share two convolutional layers before separating into different paths. The gesture classification path uses a global average pooling layer followed by two dense layers. The handedness detection path uses a global average pooling layer and a dense output layer<br/>
The last branch is responsible for the fingertips and wrist points localization task. This consists of four convolutional layers followed by an up-convolution and two convolutional layers. The final output dimensions of this block are 50×50×6. The first five channels predict the fingertip locations, and the sixth channel predicts the wrist points.<br/>
