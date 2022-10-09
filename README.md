# Face Recognition using K Nearest Neighbours and PCA

This is the assignmnt 2 of computer vision course taught by Dr. Shahzor Ahmad

The CMU Pose, Illumination, and Expression (PIE) database (http://ieeexplore.ieee.org/abstract/document/1004130/) consists of 41,368 images of 68 subjects. Each person is under 13 different poses, 43 different illumination conditions, and with 4 different expressions. Following illustrate various instances of the first subject.


![](./dataset/1.PNG)


For this project, we used a simplified version of the database which only contains 10 subjects spanningfive near-frontal poses, and there are 170 images for each individual. In addition, all the images have been resized to 32x32 pixels. Each row is an instance and each column a feature. The first 170 instances belong to the first subject, the next 170 to the second subject and so on.

* Pre-processing the dataset
* Implementation of a k Nearest Neighbours (k-NN) classifier without using in-built library
* Implementation of dimensionality reduction technique PCA in conjunction with the k-NN

