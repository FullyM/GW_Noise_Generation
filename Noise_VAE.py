import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.functional as F


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=2, stride=2)

        self.fc1 = nn.Linear(13*13, 64)
        # self.fc2 = nn.Linear(128, 64)

        self.mu_f = nn.Linear(32, 8)
        self.logstd_f = nn.Linear(32, 8)

        self.tconv1 = nn.ConvTranspose2d(8, 6, kernel_size=10)
        self.tconv2 = nn.ConvTranspose2d(6, 4, kernel_size=10, stride=3)
        self.tconv3 = nn.ConvTranspose2d(4, 3, kernel_size=10, stride=3)

    def reparametrize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mu + (eps*std)
        return z

    def sample_z(self, x, mu, logstd):
        b, l = x.shape
        x = x.view(b, 2, l // 2)
        mu = self.mu_f(x[:, 0, :])
        log_std = self.log_std_f(x[:, 1, :])

        z = self.reparametrization(mu, log_std)

        return mu, log_std, z

    def enc(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = x.flatten()
        x = self.fc1(x)
        return x

    def dec(self, z):
        b, l = z.shape
        z = z.view(b, l, 1, 1)
        z = F.relu(self.tconv1(z))
        z = F.relu(self.tconv2(z))
        z = F.relu(self.tconv3(z))
        return z


def ELBO(mse, mu, logstd):
    kl = -0.5*torch.sum(1+logstd-mu**2-2*torch.exp(logstd))
    return kl+mse


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        counter += 1
        optimizer.zero_grad()
        enc = model.forward_enc(data)
        mu, logstd, z = model.get_z(enc)
        rec = model.forward_dec(z)
        mse = F.mse_loss(rec, data, reduction='sum')
        loss = ELBO(mse, mu, logstd)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()/len(data)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()/len(data)))

    total_loss /= counter
    return total_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0.0
    counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            counter += 1
            enc = model.forward_enc(data)
            mu, logstd, z = model.get_z(enc)
            rec = model.forward_dec(z)
            mse = F.mse_loss(rec, data, reduction='sum')
            test_loss += ELBO(mse, mu, logstd).item()/len(data)
            if counter == len(test_loader.dataset) // len(target):
                recon_images = rec
                originals = data
        test_loss /= counter
    return test_loss, recon_images, originals


