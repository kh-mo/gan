import os
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(args.latent_z_dim, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 784)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.sigmoid1(x)
        return x

def save_images(image, epoch):
    saved_folder = os.path.join(os.getcwd(), "saved_image")
    try:
        os.mkdir(saved_folder)
    except FileExistsError as e:
        pass
    save_image(image, saved_folder+'/'+str(epoch+1)+'_epoch_image.png', nrow=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--latent_z_dim', type=int, default=100)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## load dataset
    train_images, train_labels = torch.load(os.path.join(os.getcwd(), "preprocess/train.pt"))
    train_images = (train_images - torch.min(train_images))/ (torch.max(train_images) - torch.min(train_images))
    train_images_loader = DataLoader(train_images.to(device=args.device), batch_size=args.batch_size)

    ## modeling
    normal_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.))
    G = Generator(args).to(device=args.device)
    D = Discriminator().to(device=args.device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)
    loss_function = torch.nn.BCELoss().to(device=args.device)

    ## training
    for epoch in range(args.epochs):
        g_loss_list = []
        d_loss_list = []
        for sample in train_images_loader:
            z = normal_distribution.sample(sample_shape=torch.Size([sample.size(0), args.latent_z_dim])).to(device=args.device)
            generated_image = G(z)

            g_loss = loss_function(D(generated_image), torch.ones_like(D(G(z))))
            g_loss_list.append(g_loss.item())
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            d_loss_fake = loss_function(D(generated_image.detach()), torch.zeros_like(D(G(z))))
            d_loss_real = loss_function(D(sample), torch.ones_like(D(G(z))))
            d_loss = (d_loss_fake + d_loss_real) / 2
            d_loss_list.append(d_loss.item())
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        z = normal_distribution.sample(sample_shape=torch.Size([25, args.latent_z_dim])).to(device=args.device)
        generated_image = G(z)
        print("epoch :", epoch, ",\t g_loss :", sum(g_loss_list)/len(g_loss_list), ",\t d_loss :", sum(d_loss_list)/len(d_loss_list))
        save_images(generated_image, epoch)


    try:
        os.mkdir(os.path.join(os.getcwd(), "saved_model"))
    except FileExistsError as e:
        pass
    torch.save(G.state_dict(), os.path.join(os.getcwd(), "saved_model/generator"))
    torch.save(D.state_dict(), os.path.join(os.getcwd(), "saved_model/discriminator"))