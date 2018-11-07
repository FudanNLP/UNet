#!/bin/bash
# File              : download.sh
# Author            : Sun Fu <cstsunfu@gmail.com>
# Date              : 29.06.2018
# Last Modified Date: 05.11.2018
# Last Modified By  : Sun Fu <cstsunfu@gmail.com>
#!/usr/bin/env bash

# Download SQuAD dataset
mkdir -p SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O SQuAD/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O SQuAD/dev-v2.0.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 -O SQuAD/elmo_weights.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json -O SQuAD/elmo_options.json

# Download GloVe
mkdir -p glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d glove

# Download Lib
pip3 install ujson
pip3 install spacy

# Download Spacy language models
python3 -m spacy download en
