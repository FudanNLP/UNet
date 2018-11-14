# UNet

The official implementation of paper [U-Net: Machine Reading Comprehension with Unanswerable Questions](https://arxiv.org/abs/1810.06638).

## Requirements

    allennlp
    pytorh 0.4
    anaconda3 for python 3.6

## For SQuAD2.0

* run '''bash download.sh''' to download the dataset and install some necessary packages.
* python prepro.py
* 'python train.py' for train
* 'python train.py --eval' for eval

The paper shows the result of this implementation.
