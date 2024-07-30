import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from dsvdd.model import autoencoder, network
from dsvdd.utils.utils import weights_init_normal
import os
import torch.nn as nn

import ipdb

class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.net = network(self.args.latent_dim).to(self.device)

        self.weights_path = f'/home/cal-05/hj/0726/yolov5/dsvdd/weights/{self.args.dataset}'
        self.pretrained_weights_path = f'{self.weights_path}/pretrained_parameters.pth'
        self.trained_weights_path = f'{self.weights_path}/trained_parameters.pth'
        self.ensure_directory_exists(self.weights_path)

    def ensure_directory_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    def pretrain(self):
        """Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)

        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        criterion = nn.MSELoss()

        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0

            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()

            avg_train_loss = total_loss / len(self.train_loader)

            print(f"Epoch {epoch + 1}/{self.args.num_epochs_ae}: Train Loss = {avg_train_loss:.3f}")

            ae.train()
        self.save_weights_for_DeepSVDD(ae, self.train_loader)

    def save_weights_for_DeepSVDD(self, model, dataloader):
        c = self.set_c(model, dataloader)

        self.net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        self.net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().numpy(), 'net_dict': self.net.state_dict()}, self.pretrained_weights_path)

    def set_c(self, model, dataloader, eps=0.1, outlier_fraction=0.1):
        model.eval()

        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())

        z_ = torch.cat(z_)
        c = self.robust_mean(z_, outlier_fraction)

        print("before")
        print(c)

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        print(c)
        return c

    def robust_mean(self, z, outlier_fraction):
        """Compute the robust mean of z by excluding outliers."""
        z_np = z.cpu().numpy()
        mean = np.mean(z_np, axis=0)
        std = np.std(z_np, axis=0)
        non_outliers = (np.abs(z_np - mean) < outlier_fraction * std)
        filtered_z = z_np[non_outliers]
        robust_mean = np.mean(filtered_z, axis=0)
        return torch.tensor(robust_mean).to(self.device)

    def train(self):
        """Training the Deep SVDD model"""

        if self.args.pretrain:
            state_dict = torch.load(self.pretrained_weights_path)
            self.net.load_state_dict(state_dict['net_dict'])
            c = torch.tensor(state_dict['center']).to(self.device)
        else:
            self.train_contrastive()
            state_dict = torch.load(self.pretrained_weights_path)
            self.net.load_state_dict(state_dict['net_dict'])
            # 중심 
            c = torch.tensor(state_dict['center']).to(self.device)

        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        self.net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            avg_train_loss = total_loss / len(self.train_loader)

            print(f'Training Deep SVDD... Epoch: {epoch + 1}/{self.args.num_epochs}, Train Loss: {avg_train_loss:.3f}')

        self.net = self.net
        self.c = c

        torch.save({'center': self.c.cpu().numpy(), 'net_dict': self.net.state_dict()}, self.trained_weights_path)
