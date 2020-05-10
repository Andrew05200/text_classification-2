### Introduction
pytorch 版本 textcnn
### Requirement
* python 3.6
* gtx1070ti  cuda10
* pytorch 1.0

### Result
```python main.py train```

result:

```
cuda is avaiable.....
load train and eval data done, time： 0:00:10
Epoch: 1
Iter: 100,      Train Loss: 1.29,       Train Acc: 0.594        Val Acc: 0.725,         Time: 0:00:07 *
Iter: 200,      Train Loss: 0.305,      Train Acc: 0.922        Val Acc: 0.841,         Time: 0:00:14 *
Iter: 300,      Train Loss: 0.332,      Train Acc: 0.938        Val Acc: 0.877,         Time: 0:00:20 *
Iter: 400,      Train Loss: 0.0996,     Train Acc: 0.969        Val Acc: 0.915,         Time: 0:00:26 *
Iter: 500,      Train Loss: 0.216,      Train Acc: 0.938        Val Acc: 0.915,         Time: 0:00:32
Iter: 600,      Train Loss: 0.229,      Train Acc: 0.922        Val Acc: 0.866,         Time: 0:00:38
Iter: 700,      Train Loss: 0.287,      Train Acc: 0.938        Val Acc: 0.9,   Time: 0:00:45
Epoch: 2
Iter: 800,      Train Loss: 0.445,      Train Acc: 0.875        Val Acc: 0.901,         Time: 0:00:51
Iter: 900,      Train Loss: 0.108,      Train Acc: 0.938        Val Acc: 0.919,         Time: 0:00:57 *  
...

```

todo:

1、完成测试和预测代码


### Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

