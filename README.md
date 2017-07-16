This is a tensorflow implementation of  NLPCC2017 DBQA task. Our result ranks 5th amoung the 21 submission.

[Enhanced Embedding based Attentive Pooling Network for Answer Selection](http://tcci.ccf.org.cn/conference/2017/)

We utilize chinese wiki corpus to train our embedding. You can train embedding by youself or contact us to get what we use.

## Requirements

-python2.7

-Tensorflow = 1.2

-gensim

-numpy

## Training


```
./train.py --overlap_needed True --position_needed True
```

##



| method | pooling | map(test1) | map(test2)
| :--- | :----: | ----: |:----:|
| CNN-base | max | 0.782 | 0.657
| CNN-base | attentive | 0.772  | 0.646
| +overlap | max | 0.828 | 0.674
| +overlap | attentive | 0.811  | 0.672|
| +position,overlap | attentive | 0.819 | 0.675
| +position,overlap |  max | 0.834  | 0.679


