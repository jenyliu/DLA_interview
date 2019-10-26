# DLA_interview

## Introduction
The objective of this project is to learn more about conditional generative models. Having worked with GANs, it seems beneficial to study more about adding additional descriptive information with the input image to produce models that are able to distinctly represent specific subjects in the generated data. It seems to be a part of how users can select specific features or labels for the model to generate. As an early step of looking at this and taking into account the limitations of resources and time, this project will be experimenting with the vanilla variational autoencoder and a conditional variational autoencoder.

## Process
The original Variational Autoencoder paper and code implemeted in pytorch and the accompanying paper which is initially applied to the MNIST. Since MNIST is a dataset that has been implemented many times and the different classes can be identified with only a few pixels, the variational autoencoder will also be applied to the FashionMNIST data and KMNIST data to have a better understanding of performance.

Original VAE paper and accompanying code example
https://arxiv.org/pdf/1312.6114.pdf

https://github.com/pytorch/examples/tree/master/vae


# Installation changed:
Install pytorch from here:
https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/

For the Jetson TX2, Pytorch cannot be installed using the method described on the above Github page. An alternate version for the GPU must be downloaded from the NVIDIA site as indicated by the link above. 

## Process
The original Variational Autoencoder paper and code implemeted in pytorch and the accompanying paper. Originally implemented on MNIST.

https://github.com/pytorch/examples/tree/master/vae

https://arxiv.org/pdf/1312.6114.pdf

To implement a conditional variational autoencoder, the input size of the encoder neural network is increased by the number of labels. The digit label is one hot encoded and concatenated to the initial input size of 28 * 28 = 784. The number of hidden nodes and the output size of the encoder network is left as is. The input size of the decoder network is the latent variable size (output of encoder) plus the label. The resulting output should be equivalent to the size of the image.

Since the original






Adjusted original VAE to use FashionMNISt, appears to generate images that are recognizable. What does changing the size of the latent space and increasing the training length do?

## Comments

One thing I noticed about VAE is that it works well on MNIST but fails to work on others, there's a high level of compression in VAE since it encodes a high dimensional data space to a lower dimensional Gaussian. This method works well for MNIST since the labels can be identified from as little as one pixel, as demonstrated here: https://gist.github.com/dgrtwo/aaef94ecc6a60cd50322c0054cc04478.

Will also check performance on FashionMNIST, less used data set.

## Results

Ideally, I would like to look into Real NVP at https://arxiv.org/pdf/1605.08803.pdf and potentially generating labeled images since it produces sharper images and seems to be a method that creates images with the same level of sharpness as Generative Adversarial Networks, without being quite as sensitive to hyperparameters.
