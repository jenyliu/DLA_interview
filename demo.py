from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

import math

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
tr_split_len = 6000
te_split_len = 1000

tr = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
te = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())

part_tr = torch.utils.data.random_split(tr, [tr_split_len, len(tr)-tr_split_len])[0]
part_te = torch.utils.data.random_split(te, [te_split_len, len(te)-te_split_len])[0]

train_loader = torch.utils.data.DataLoader(part_tr, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(part_te, batch_size=args.batch_size, shuffle=True, **kwargs)



train_lossList = []
test_lossList = []
train2_lossList = []
test2_lossList = []

def idx2onehot(idx,n=10):

    onehot = torch.zeros(idx.size(0),n)
    onehot.scatter_(1,idx.data, 1)

    return onehot

class VAE(nn.Module):


    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, 2)
        self.fc22 = nn.Linear(500, 2)
        self.fc3 = nn.Linear(2, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        #x = torch.cat((x,y), dim=1)
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        #z = torch.cat((z, y), dim=1)
        return self.decode(z), mu, logvar

class CVAE(nn.Module):


    def __init__(self):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(794, 500)
        self.fc21 = nn.Linear(500, 2)
        self.fc22 = nn.Linear(500, 2)
        self.fc3 = nn.Linear(12, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        x = torch.cat((x,y), dim=1)
        mu, logvar = self.encode(x.view(-1, 794))
        z = self.reparameterize(mu, logvar)
        z = torch.cat((z, y), dim=1)
        return self.decode(z), mu, logvar


model1 = CVAE().to(device)
model2 = VAE().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    #BCE = F.mse_loss(recon_x, x.view(-1, 3072), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #MMD = compute_mmd(x,y)
    return BCE + KLD
    #return BCE + HD


def train(epoch):
    model1.train()
    model2.train()
    train_loss = 0
    train_loss2 = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.view(-1,784)
        data = data.to(device)

        label = idx2onehot(label.view(-1,1))
        label = label.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        recon_batch, mu, logvar = model1(data,label)
        recon_batch2, mu2,logvar2 = model2(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss2 = loss_function(recon_batch2, data, mu2, logvar2)
        loss.backward()
        loss2.backward()
        train_loss += loss.item()
        train_loss2 += loss2.item()
        optimizer1.step()
        optimizer2.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    train_lossList.append(train_loss / len(train_loader.dataset))
    train2_lossList.append(train_loss2 / len(train_loader.dataset))


def test(epoch):
    model1.eval()
    model2.eval()
    test_loss = 0
    test_loss2 = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.view(-1,784)
            data = data.to(device)

            label = idx2onehot(label.view(-1,1))
            label = label.to(device)

            recon_batch, mu, logvar = model1(data,label)
            recon_batch2, mu2, logvar2 = model2(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            test_loss2 += loss_function(recon_batch2, data, mu2, logvar2).item()

            if i == 0:
                n = min(data.size(0), 8)
                #true = torch.tensor(data)
                #true = true*2 + 0.5
                comparison = torch.cat([data.view(args.batch_size, 1, 28, 28)[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n],
                                      recon_batch2.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    test_loss2 /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    test_lossList.append(test_loss)
    test2_lossList.append(test_loss2)

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(80, 2).to(device)
            x = np.array([0,1,2,3,4,5,6,7,8,9])
            y = np.repeat(x,8)
            y = np.reshape(y,(80,1))
            y = torch.tensor(y).to(dtype=torch.long)
            y = idx2onehot(y).to(device,dtype=sample.dtype)
            sample2 = sample.clone().detach()
            sample = torch.cat((sample, y),dim=1)
            sample = model1.decode(sample).cpu()
            sample2 = model2.decode(sample2).cpu()
            save_image(sample.view(80, 1, 28, 28),
                       'results/VAE_sample_' + str(epoch) + '.png')
            save_image(sample2.view(80, 1, 28, 28),
                       'results/CVAE_sample_' + str(epoch) + '.png')
    epochs_list = range(1, args.epochs + 1)
    plt.plot(epochs_list, train2_lossList, "r", epochs_list, test2_lossList, "r--",
         epochs_list, train_lossList, "b", epochs_list, test_lossList, "b--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Function - MNIST")
    plt.legend(["Train-VAE", "Test-VAE", "Train-CVAE", "Test-CVAE"])
    plt.show()
