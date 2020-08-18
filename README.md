
# FixMatch

This is my implementation of the experiment in the paper of [fixmatch](https://arxiv.org/abs/2001.07685). 
I only implemented experiements on cifar-10 dataset without CTAugment.


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

To train the model with 40 labeled samples, you can run the script: 
```
    $ python train.py --n-labeled 40 
```
where `40` is the number of labeled sample during training.


## Results
After training the model with 40 labeled samples for 5 times with the command:
```
    $ python train.py --n-labeled 40 
```
I observed top-1 accuracy like this:  

| #No. | 1 | 2 | 3 | 4 | 5 |
|:---|:---:|:---:|:---:|:---:|:---:|
|acc | 91.81 | 91.29 | 89.51 | 91.32 | 79.42 |


Note: 
1. There is no need to add interleave, since interleave is used to avoid the bias of bn status. MixMatch uses interleave because because they run forward computation with three data batches for three times, if you combine the three batches together and run only one pass of forward computation with the combined batch, the results should be same. You may refer to my implementation of mixmatch [here](https://github.com/CoinCheung/mixMatch.git), which does not use interleave and still achieves similar results. 

2. There are two methods to deal with the buffers in the operation of ema: One is directly copying them as the ema buffer states and the other is implementing ema on these buffer states. Generally speaking, there should not be a large gap between these two methods, since in the model of resnet, the buffers are the `running_mean/running_var` of the `nn.BatchNorm` layers. During training process, these BN buffers are updated with the moving average method which is same as what the `ema` operator does. What the `ema` operator does is to estimate the expectation of the associated parameters by smoothing a series of values. Ema operation can be seen as averaging the most recent values of the series, and by averaging the values, we are computing the less noised parameter value(which can simply be treated as the expectation). To directly copy the buffers, we are using the `one-order` smoothing, while to implement ema on them, we are using the `two-order` smoothing. In general, we can have a good enough expected value with one-order smoothing, though two-order smoothing should be more unbiased.

3. The method based on naive random augmentation will cause a relatively large variance. If you set random seed free, and generate the split of labeled training set randomly each time, you may observe that the validation accuracy would fluctuate within a big range. In the paper, the authors used CTAugment which introduced some feedback to the data augmentation strategy, which will reduce the variance.

