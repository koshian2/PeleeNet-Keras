# PeleeNet-Keras
An Unoffical Implementation of PeleeNet by TensorFlow, Keras.  
Implemented training with CIFAR-10.

## Original Paper
R. J. Wang, X. Li, C. X. Ling. *Pelee: A Real-Time Object Detection System on Mobile Devices*. NIPS. 2018.
[https://arxiv.org/abs/1804.06882](https://arxiv.org/abs/1804.06882)

The original implementation by Caffe.
[https://github.com/Robert-JunWang/Pelee](https://github.com/Robert-JunWang/Pelee)

## Get started
```python
from pelee_net import PeleeNet
model = PeleeNet(input_shape=(224,224,3), use_stem_block=True, n_classes=1000)
```

### Parameters
* **input_shape** : Resolution of input images. 224x224x3 by default(same as the original).
* **use_stem_block** : Whether to use Stem Block. If True it's same as the original, if False 
input is connected directly to the first Dense Layer.
* **n_classes** : Number of classes in prediction

## Results on CIFAR-10

| Augmentation | Stem Block | No weight Decay | 5e-4 Weight Decay |
|:------------:|:----------:|----------------:|------------------:|
|      No      |     No     |          0.7633 |            0.9247 |
|      No      |     Yes    |          0.8280 |            0.9218 |
|      Yes     |     No     |          0.8881 |            0.9446 |
|      Yes     |     Yes    |          0.8996 |            0.9410 |

* Enable stem block cases : Input=(224, 224, 3), Upsampling x7
* Disable stem block cases : Input=(32, 32, 3), No upsampling

Data augmentation is the standard data augmentation(4 pixels shift and horizontal flip).

### No weight decay
![](https://github.com/koshian2/PeleeNet-Keras/blob/master/images/cifar10_no_weight_decay.png)

### 5e-4 weight decay
![](https://github.com/koshian2/PeleeNet-Keras/blob/master/images/cifar10_5e-4_weight_decay.png)

There was no discussion int the paper on weight decay. But I noticed that weight decay is important in increasing the accuracy of CIFAR-10, so I added it.

## Details (Japanese)
DenseNetの軽量版、PeleeNetをKerasで実装した  
[https://qiita.com/koshian2/items/187e240f478504079e7a](https://qiita.com/koshian2/items/187e240f478504079e7a)
