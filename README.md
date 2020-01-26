
# MixMatch

This is my implementation of the experiment in the paper of [mixmatch](https://arxiv.org/abs/1905.02249). On my platform, the accuracy reaches 89+ on cifar10 with 250 labeled images.


## Environment setup

* 2080ti gpu
* ubuntu-16.04
* python3.6.9
* pytorch-1.2.0 from conda
* cudatoolkit-10.0.130 from conda
* cudnn-7.6.2 in /usr/lib/x86_64-linux-gpu


## Dataset
download cifar-10 dataset: 
```
    $ mkdir -p dataset && cd dataset
    $ wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    $ tar -xzvf cifar-10-python.tar.gz
```

## Train the model
```
    $ sh run.sh
```


## Some notes I made during experiment

1. use exponential moving average (EMA) to update model parameters.

2. though softmax has negative impact on the training with mse loss, the paper still use mse loss (from softmax predictions of unlabeled data to guessed label) to train the model.

3. it is better to warmup the balance factor between the labeled loss and the unlabeled loss. The official repository let the factor improve from 0 to 75 during the whole 1024 epoches. Maybe it is better to slowly increase the contribution of the unlabeled data.

4. do not use dropout in the wide-resnet-28-2.

5. wd should be added to model(not ema) weight directly rather than added via optimizer options, which is actually added to the gradients.

6. ~~use ema parameters to guess the labels.~~ That is what mean teacher does.

7. mixup should use different mix coefficients for each samples in the batch, rather than one coefficient for the whole batch.
