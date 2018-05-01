#!/bin/bash

set -ex

git clone https://github.com/tensorflow/tensorflow 
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
wget https://github.com/bazelbuild/bazel/releases/download/0.12.0/bazel-0.12.0-installer-linux-x86_64.sh
chmod a+x bazel-0.12.0-installer-linux-x86_64.sh 
sudo ./bazel-0.12.0-installer-linux-x86_64.sh
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
cd tensorflow 
echo -e '\n\n/usr/lib/python3/dist-packages\n\n\nn\nn\n\n\n\n\n\n\n\n\n\n' | ./configure 
bazel build --config=opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo -H pip3 install /tmp/tensorflow_pkg/tensorflow-1.8.0rc1-cp35-cp35m-linux_x86_64.whl
python -c "import tensorflow as tf; hello = tf.constant('Hello, TensorFlow'); sess = tf.Session(); print(sess.run(hello));"
