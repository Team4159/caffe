# Team 4159 Reflective Tape Detector

This is a CNN (convolutional neural network) for detecting reflective tape in FIRST Steamworks. It uses Single-Shot Detection forked from [weiliu89/caffe](https://github.com/weiliu89/caffe/tree/ssd/).

## Installation
tl;dr:
Create a fresh install of Ubuntu 17.04.
Install packages. Install GCC 6.4 instead of 7 to avoid having to build and install 6.4
Install CUDA 9 (or the correct OpenCL distribution if you have AMD GPU)
Config and Build SSD Caffe

1. Install the following packages:
```Shell
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libatlas-base-dev python-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev python-numpy gcc-6.4 bison curl flex g++-multilib gcc-multilib git gperf lib32ncurses5-dev lib32readline-dev lib32z1-dev libesd0-dev liblz4-tool libncurses5-dev libsdl1.2-dev libssl-dev libwxgtk3.0-dev libxml2-utils lzop pngcrush schedtool xsltproc python-skimage python-pip
sudo pip install protobuf
```

2. Install CUDA 9 (download from nvidia, nvidia-cuda-toolkit has version 8). This requires GCC 6.x (6.4), because GCC 5.3 will NOT work on Ubuntu 17.10 (and I presume 17.04): 
https://stackoverflow.com/questions/39130040/cmath-hides-isnan-in-math-h-in-c14-c11
http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation

3. Building SSD Caffe:
* Copy the Makefile.config from setup to the root folder
* Fix error of CUDA version in boost. Copy the nvcc.hpp in setup to /usr/include/boost/config/compiler/nvcc.hpp

Now build (jN refers to build with N cores, replace with the number of cores for your computer:
```Shell
make all -j8
make py -j8
make test -j8
```

## Preparing Data

First set the CAFFE_ROOT environment variable in your .bashrc file to the root installation folder and run `source ~/.bashrc` to apply.

Clone our caffe-data repo inside the data folder and follow the instructions for that repo to get the data. Now you can prepare the LMDB:

```Shell
cd $CAFFE_ROOT
./data/ml/create_data.sh
```

## Training

Train the model using `examples/ssd/ssd_pascal_steamworks.py`:

```Shell
cd $CAFFE_ROOT
python examples/ssd/ssd_pascal_steamworks.py
```

It's very likely that won't do anything, so far due to I/O issues. To verify your GPU works with Caffe, follow the mnist example to training the LeNet model, which should work.
