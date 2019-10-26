# DLA_interview

## Introduction
The objective of this project is to learn more about conditional generative models. Having worked with GANs, it seems beneficial to study more about adding additional descriptive information with the input image to produce models that are able to distinctly represent specific subjects in the generated data. It seems to be a part of how users can select specific features or labels for the model to generate. As an early step of looking at this and taking into account the limitations of resources and time, this project will be experimenting with the vanilla variational autoencoder and a conditional variational autoencoder.


## Installation changed:
Install pytorch from here:
https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/

For the Jetson TX2, Pytorch cannot be installed using the method described on the above Github page. An alternate version for the GPU must be downloaded from the NVIDIA site as indicated by the link above. 

## Process
The original Variational Autoencoder paper and code implemeted in pytorch and the accompanying paper which is initially applied to the MNIST. Since MNIST is a dataset that has been implemented many times and the different classes can be identified with only a few pixels, the variational autoencoder will also be applied to the FashionMNIST data and KMNIST data to have a better understanding of performance.

Original VAE paper and accompanying code example
https://arxiv.org/pdf/1312.6114.pdf

https://github.com/pytorch/examples/tree/master/vae

The paper on the conditional variational autoencoder and it's loss function is as follows
https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf

To implement a conditional variational autoencoder, the original varaiational autoencoder is modified several ways:

-The input size of the encoder neural network is increased by the number of labels. The digit label is one hot encoded and concatenated to the initial input size of 28 * 28 = 784 so the input of the encoder network is 28 * 28 + 10 - 794.
-The input size of the decoder neural network is increased by the number of labels. For the original MNIST network a latent variable size of 2 was chosen, so the input to the decoder network is now 2 + 10 = 12.
-All additional layers and nodes of the networks remain the same

The CVAE network is further modified to have the label data concatenated to the inputs and the reparametrized latent variables. The loss function is still calculated over the same features and does not change with label data.

All experiments were run for 200 epochs with a learning rate of 0.001. The hidden layer on the encoder network and the decoder network have 500 nodes as described in the variational autoencoder paper for experiments done on the MNIST data. Changing the number of nodes did not seem to make any discernible difference on the images as seen by a person, so there was no need to adjust. The latent variable size for the MNIST data set was set to 2. Since the KMNIST and FashionMNIST datasets had more detail, the latent variable size was adjusted to 10.

A few experiments were run on the CIFAR10 data set as well, but due to poor performance of variational autoencoders on these images, those results were not pursued in this project. 


## Results
The conditional variational autoencoder always prints out the correct digit or article of clothing for the FashionMNIST data. This is likely becasue the label data is encoded in the input of the encoder. When the latent space is generated, it enocdes each digit as a separate Gaussian function where Z~N(0,I). In the vanilla variational autoencoder, all digits are encoded to the same Z~N(0,I), where different digits are clustered. This makes points that line near the boundaries of different digits less discernible. When checking even later samples of reconstructed test points, examples of digits that differ in value can be seen.

Due to the latent variables selected in the KL loss and the reconstruction loss, the loss for the conditional variational autoencoder is also lower than the variational autoencoder. 
Ideally, I would like to look into Real NVP at https://arxiv.org/pdf/1605.08803.pdf and potentially generating labeled images since it produces sharper images and seems to be a method that creates images with the same level of sharpness as Generative Adversarial Networks, without being quite as sensitive to hyperparameters.
