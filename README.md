[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# milliEgo
### [Project](https://christopherlu.github.io/publications/milliego) | [Youtube](https://www.youtube.com/watch?v=I9vjoKGY2ts&feature=youtu.be) | [Paper](https://christopherlu.github.io/files/papers/[SenSys2020]milliEgo.pdf) <br>

Simplified [docker](https://www.docker.com/) version for the implementation of our 6-DOF Egomotion Estimation method via a single-chip mmWave radar ([TI AWR1843](https://www.ti.com/product/AWR1843)) and a commercial-grade IMU. Our method is the first-of-its-kind DNN based odometry approach that can estimate the egomotion from the sparse and noisy data returned by a single-chip mmWave radar. <br><br>
[milliEgo: Single-chip mmWave Aided Egomotion Estimation with Deep Sensor Fusion](https://christopherlu.github.io/publications/milliego)  
Chris Xiaoxuan Lu, Muhamad Risqi U. Saputra, Peijun Zhao, Yasin Almalioglu, Pedro P. B. de Gusmao, Changhao Chen, Ke Sun, Niki Trigoni, Andrew Markham
In [SenSys 2020](https://www.sigmobile.org/sensys/2020/).  

## Prerequisites
- Linux
- Docker
- Python 3.6.8
- CUDNN 9.0

## Getting Started
### Docker Installation
Make a tensorflow 1.9.0 docker environment. Install nvidia-docker with https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
Simply install docker on your machine and pull the correct version of tensorflow docker:

```
docker pull tensorflow/tensorflow:1.9.0-gpu-py3
```
### Pre-trained mmWave Radar Feature Extractor and milliEgo model
- After git clone this repository, enter the project directory,
```
mkdir -p models/cross-mio
```
- Download the pre-trained CNN model ['cnn.h5'](https://www.dropbox.com/s/osi5w1gaaiiykhi/cnn.h5?dl=0) (dropbox link) for mmWave feature extraction and put it under `./models/`
- Download the trained milliEgo model '140' and the respective config file `nn_opt.json` from [here](https://www.dropbox.com/sh/g0rpk0ah6oldyp9/AACTjB6fIUfw02ol3Adj2wqga?dl=0) (dropbox link). Put both of them in `./models/cross-mio/`.

### Dataset
- To train and test a model, please download our dataset from [here](https://www.dropbox.com/s/ap54f319vpttaat/dataset.zip?dl=0) (dropbox link).

- After downloading and unzip, please put the dataset folder in `<host dataset dir path>` of your host machine.

### Start the docker container

Suppose dataset is stored in host machine under `<host dataset dir path>`. Run docker with:

```
docker run --gpus all -it --rm -v <host dataset dir>:/datasets/multi_gap_5 tensorflow/tensorflow:1.9.0-gpu-py3 bash
```
Note the 'multi_gap_5' is dummy directory name but essentially it implies the down-sampling intervals of the mmWave radar data - you should have enough parallax for a good visual odometry.

### Install dependencies in docker container:

```
pip install tensorflow-estimator==1.14.0
pip install keras==2.1.6
apt-get update -y
apt-get install python3-tk
```

### Testing
In host machine, copy code folder into docker container.

```
docker cp <host code dir> <container ID>:/code
```

In the docker container, run testing on pre-trained model (cross-attention):

```
cd /code
python test_trajectory.py
```
Check the generated trajectories in `/code/figs` and results to be quantitatively evaluated in `/code/results`.

### Training
In docker container, run training:

```
python /code/train_cross_att.py
```

## Citation

If you find this useful for your research, please use the following.

```
@inproceedings{lu2020milliego,
  title={milliEgo: single-chip mmWave radar aided egomotion estimation via deep sensor fusion},
  author={Lu, Chris Xiaoxuan and Saputra, Muhamad Risqi U and Zhao, Peijun and Almalioglu, Yasin and de Gusmao, Pedro PB and Chen, Changhao and Sun, Ke and Trigoni, Niki and Markham, Andrew},
  booktitle={Proceedings of the 18th Conference on Embedded Networked Sensor Systems (SenSys)},
  year={2020}
}
```
