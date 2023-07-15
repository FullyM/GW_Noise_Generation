import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FirstVAE(nn.Module):
    def __init__(self):
        super(FirstVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(4*4*40, 128)

        self.mu_f = nn.Linear(64, 16)
        self.logstd_f = nn.Linear(64, 16)

        self.tconv1 = nn.ConvTranspose2d(16, 8, kernel_size=10)
        self.tconv2 = nn.ConvTranspose2d(8, 4, kernel_size=8, stride=4)
        self.tconv3 = nn.ConvTranspose2d(4, 3, kernel_size=5, stride=3, padding=3)

    def reparametrize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mu + (eps*std)
        return z

    def sample_z(self, x):
        b, l = x.shape
        x = x.view(b, 2, l // 2)
        mu = self.mu_f(x[:, 0, :])
        log_std = self.logstd_f(x[:, 1, :])

        z = self.reparametrize(mu, log_std)

        return mu, log_std, z

    def enc(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x

    def dec(self, z):
        b, l = z.shape
        z = z.view(b, l, 1, 1)
        z = F.relu(self.tconv1(z))
        z = F.relu(self.tconv2(z))
        z = F.relu(self.tconv3(z))
        return z

    def forward(self, x):
        x = self.enc(x)
        mu, logstd, z = self.sample_z(x)
        x = self.dec(z)



