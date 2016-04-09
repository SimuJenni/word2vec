# R&D Research Project

This repo contains three implementations of word2vec.
It also contains a folder dedicated to the LATEX-report and a data folder named word2vec_data where all the training data, temporary files and results shold be stored.

### Original C Implementation

Just build using make and run the demo-scripts.
Code is highly optimized and therefore difficult to modify! But should be fine for experiments.

### Gensim Python Implementation

A nice re-implementation of the original C-code in Python. Well documented and readable. Code is well optimized and performs as well as the C code.

Imo, prefered implementation for running experiments.

https://radimrehurek.com/gensim/models/word2vec.html

### Tensorflow Implementation

Implementation of word2vec in Googles TensorFlow framework. Probably easiest to modify.

https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html
