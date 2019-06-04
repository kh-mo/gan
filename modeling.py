import torch
import argparse
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(args.latent_z_dim, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 784)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--latent_z_dim', type=int)
    args = parser.parse_args()

    ## load dataset

    ## modeling
    normal_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.))
    z = normal_distribution.sample(sample_shape=torch.Size([args.batch_size, args.latent_z_dim]))
    G = Generator(args)
    D = Discriminator()

    ## training
    for epoch in args.epochs:
        args.batch_size
