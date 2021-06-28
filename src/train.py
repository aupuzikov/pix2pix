import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from model import generator, discriminator
from utils import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(g: nn.Module, d: nn.Module, dataloader, optims, num_epochs=15):
    g = g.to(device)
    d = d.to(device)
    from collections import defaultdict

    losses = defaultdict(list)
    for i in tqdm(range(num_epochs), desc='Epochs'):
        generated = None
        for data in tqdm(dataloader, position=0, leave=True, desc=f'epoch {i}, batches'):
            images = data[0]
            borders = images[:, :, :, :256].to(device)
            real = images[:, :, :, 256:].to(device)
            generated = g(borders)

            discriminator_on_generated = d(borders, generated.detach())
            discriminator_on_real = d(borders, real)
            #         print('discr', discriminator_on_generated.shape)
            discriminator_loss = nn.BCELoss()

            # discriminator on real
            loss = discriminator_loss(discriminator_on_real, torch.ones(discriminator_on_real.shape).to(device))
            # discriminator on fakes
            loss += discriminator_loss(discriminator_on_generated,
                                       torch.zeros(discriminator_on_generated.shape).to(device))
            torch.sum(loss).backward()
            losses['discriminator'].append(torch.sum(loss).item())
            optims['discriminator'].step()
            optims['discriminator'].zero_grad()

            discriminator_on_generated = d(borders, generated)
            generator_loss = nn.BCELoss()
            # L1 loss
            loss = 100 * torch.abs(real - generated)

            loss += generator_loss(discriminator_on_generated, torch.ones_like(discriminator_on_generated))
            torch.sum(loss).backward()
            losses['generator'].append(torch.sum(loss).item())
            optims['generator'].step()
            optims['generator'].zero_grad()

        utils.save_image_batch(generated.detach().cpu(), rows=8, hspace=0.1, name=f'./data/image_{i}.png')


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print('working with ' + str(device))
    parser = argparse.ArgumentParser(description='model training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_path", required=True, help="path to dataset ")

    parser.add_argument("--num_epochs", required=False, default=10, type=int,
                        help="number of epochs to train model")
    parser.add_argument("--batch_size", required=False, default=4, type=int,
                        help="batch size for gradient descend")

    args = parser.parse_args()
    dataset = ImageFolder(args.dataset_path,
                          transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    gen = generator.Generator()
    dis = discriminator.Discriminator()

    optimizers = {'generator': optim.AdamW(gen.parameters(), betas=(0.5, 0.9999)),
                  'discriminator': optim.AdamW(dis.parameters(), betas=(0.5, 0.9999))}
    train(gen, dis, dataloader=loader, optims=optimizers, num_epochs=args.num_epochs)
    torch.save(gen.state_dict(), './data/generator.pt')
