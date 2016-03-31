# face-ver

This  code implements siamese network to verify face using Tensorflow. The model architecture is similiar to Caffe's example of Siamese Network, and the same loss function is custom built in the code.

To run the code : 

```shell
1. Install Docker
2. cd to the root directory of this repository
3. $docker build -t face-ver .
4. $docker run -it -v /path/to/repo/face-ver/src:/workspace/src
5. (docker)$cd workspace
6. (docker)$cd src
7. (docker)&cd serving/tensorflow
8. (docker)$./configure
9. (docker)$cd ../../face_ver
10. (docker)$bazel build :all -j 2
11. Download and extract att_faces dataset at current folder
12. (docker)$dataset.py
13. (docker)$bazel-bin/cifar10_train
14. (docker)$bazel-bin/cifar10_eval
15. let 'abc000001' be output of $ls /tmp/serving
15. (docker)$bazel-bin/inference --port=9000 /tmp/serving/abc000001
16. put any two images, img1.pgm and img2.pgm in folder 'test'
17. (docker)$bazel-bin/tensorflow_serving/example/client --server=localhost:9000
18. The output printed will be the distance between two feature vector. Lesses the distance, more is the probability of images being of same face.
```
