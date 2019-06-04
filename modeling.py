import os
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
        self.layer1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--latent_z_dim', type=int, default=100)
    args = parser.parse_args()

    ## load dataset
    train_images, train_labels = torch.load(os.path.join(os.getcwd(), "preprocess/train.pt"))
    test_images, test_labels = torch.load(os.path.join(os.getcwd(), "preprocess/test.pt"))

    ## modeling
    normal_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.))
    G = Generator(args)
    D = Discriminator()

    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=1e-4)

    ## training
    for epoch in range(args.epochs):
        z = normal_distribution.sample(sample_shape=torch.Size([args.batch_size, args.latent_z_dim]))
        generated_image = G(z)
        D(real_image, D(z))

        g_loss = 10
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()