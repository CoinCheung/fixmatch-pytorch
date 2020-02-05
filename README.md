
# FixMatch

This is my implementation of the experiment in the paper of [fixmatch](https://arxiv.org/abs/2001.07685). 


## Environment setup

My platform is: 
* 2080ti gpu
* ubuntu-16.04
* python3.6.9
* pytorch-1.3.1 installed via conda
* cudatoolkit-10.1.243 
* cudnn-7.6.3 in /usr/lib/x86_64-linux-gpu


## Dataset
download cifar-10 dataset: 
```
    $ mkdir -p dataset && cd dataset
    $ wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    $ tar -xzvf cifar-10-python.tar.gz
```

## Train the model
The configurations are in `train.py`. By default, it trains with 40 labeled images. If you would like to try training with other number of labeled images, you can modify the variable of `n_labeled` in `train.py`.   

To train the model, you can run the script: 
```
    $ sh run.sh
```

Note that currently I only implemented the experiments on cifar10 with `RA` as strong augmentation settings.
