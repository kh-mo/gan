import os
import torch
import argparse
from modeling import Generator, Discriminator
from torchvision.utils import save_image

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_z_dim', type=int, default=100)
    parser.add_argument('--num_images', type=int, default=1)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(args).to(device=args.device)
    D = Discriminator().to(device=args.device)

    G.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_model/generator")))
    D.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_model/discriminator")))

    normal_distribution = torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.))

    saved_folder = os.path.join(os.getcwd(), "evaluated_image")
    try:
        os.mkdir(saved_folder)
    except FileExistsError as e:
        pass

    for i in range(args.num_images):
        z = normal_distribution.sample(sample_shape=torch.Size([1, args.latent_z_dim])).to(device=args.device)
        save_image(G(z), saved_folder + '/evaluated_' + str(i+1) + ' images.png')
    print("image generated done.")