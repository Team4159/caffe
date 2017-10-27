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
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libatlas-base-dev python-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev python-numpy gcc-6 bison curl flex g++-multilib gcc-multilib git gperf lib32ncurses5-dev lib32readline-dev lib32z1-dev libesd0-dev liblz4-tool libncurses5-dev libsdl1.2-dev libssl-dev libwxgtk3.0-dev libxml2-utils lzop pngcrush schedtool xsltproc python-skimage python-pip
sudo pip install protobuf
```

2. Install CUDA 9 (download from nvidia, nvidia-cuda-toolkit has version 8). This requires GCC 6.x (6.4), because GCC 5.3 will NOT work on Ubuntu 17.10 (and I presume 17.04): 
https://stackoverflow.com/questions/39130040/cmath-hides-isnan-in-math-h-in-c14-c11
http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation

3. Add environment variables to ~/.bashrc, then run ```source ~/.bashrc``` to persist them:
```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CAFFE_ROOT=/home/<YOUR_USERNAME>/caffe
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
```

4. Building SSD Caffe:
* Copy the Makefile.config from setup to the root folder
* Fix error of CUDA version in boost. Copy the nvcc.hpp in setup to /usr/include/boost/config/compiler/nvcc.hpp

Now build (jN refers to build with N virtual cores, replace with the number of cores for your computer):
```Shell
make -j8
make py -j8
make test -j8
```

## Preparing Data

Follow the directions in our [data repo](https://github.com/Team4159/caffe-data) to prepare our data to be trained.

## Training
Find and open examples/ssd/ssd_pascal_steamworks.py, and look for the base_lr and batch_size, and accum_batch_size variables. The learning rate determines how fast the training will advance in regards of optimizing the CNN's accuracy. The slower the learning rate, the more accurate it will be, but the faster the learning rate the more likely it can overshoot and diverge and thus making the model useless. This is indicated by a loss of NaN in the log. So how can we prevent this? By adjusting the base_lr (base learning rate). 

Also be aware of high memory usage during training. If the batch_size is too high (more than 4) then the program will crash, too low and the learning may diverge. So far 4 seems to work best for a 3GB GeForce 1060 GTX, and the accum_batch_size should be 16 to compensate for that. There also needs to be enough memory for testing while training, or it can be skipped by setting test_interval higher than max_iter.

Finally, train the model. This will take some time:

```Shell
cd $CAFFE_ROOT
python examples/ssd/ssd_pascal_steamworks.py
```
## Verification
You can evaluate a model by editing score_ssd_pascal.py to use the newly created model, and also follow [these directions](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) in order to visualize detection.

TODO How to use OpenCV to detect images in real time
