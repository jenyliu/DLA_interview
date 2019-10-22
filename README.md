# DLA_interview

## Introduction
Something that I looked at in my lab at and didn't get an opportunity to implement are variational autoencoders. I also looked at GANs while I was doing my internship at the Air Force Research Lab. When I looked at the original Variational Autoencoder I saw examples of data being reconstructed and generated randomly, with it occasionally being generated on a manifold. I want to look into generating data while selecting the class label ahead of time.

## Process
The original Variational Autoencoder paper and code implemeted in pytorch.

https://github.com/pytorch/examples/tree/master/vae

https://arxiv.org/pdf/1312.6114.pdf

## Comments

One thing I noticed about VAE is that it works well on MNIST but fails to work on others, there's a high level of compression in VAE since it encodes a high dimensional data space to a lower dimensional Gaussian. This method works well for MNIST since the labels can be identified from as little as one pixel, as demonstrated here: https://gist.github.com/dgrtwo/aaef94ecc6a60cd50322c0054cc04478.

## Results

Ideally, I would like to look into Real NVP at https://arxiv.org/pdf/1605.08803.pdf and potentially generating labeled images since it produces sharper images and seems to be a method that creates images with the same level of sharpness as Generative Adversarial Networks, without being quite as sensitive to hyperparameters.
