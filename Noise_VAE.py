import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(30)
        self.bn4 = nn.BatchNorm2d(40)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.drop3 = nn.Dropout(p=0.1)
        self.drop4 = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(4*4*40, 128)
        # self.fc2 = nn.Linear(128, 64)
        self.bn_lin1 = nn.BatchNorm1d(128)
        self.drop_lin1 = nn.Dropout(p=0.1)

        self.mu_f = nn.Linear(64, 32)
        self.logstd_f = nn.Linear(64, 32)

        self.tconv1 = nn.ConvTranspose2d(32, 20, kernel_size=10)  # 10
        self.tconv2 = nn.ConvTranspose2d(20, 10, kernel_size=8, stride=3)  # 35
        self.tconv3 = nn.ConvTranspose2d(10, 5, kernel_size=5, stride=2, padding=4)  # 65
        self.tconv4 = nn.ConvTranspose2d(5, 3, kernel_size=2, stride=2, padding=1)  # 128
        self.bn5 = nn.BatchNorm2d(20)
        self.bn6 = nn.BatchNorm2d(10)
        self.bn7 = nn.BatchNorm2d(5)
        self.drop5 = nn.Dropout(p=0.1)
        self.drop6 = nn.Dropout(p=0.1)
        self.drop7 = nn.Dropout(p=0.1)

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop4(x)
        x = x.flatten(start_dim=1)
        x = self.bn_lin1(self.fc1(x))
        x = self.drop_lin1(x)
        return x

    def dec(self, z):
        b, l = z.shape
        z = z.view(b, l, 1, 1)
        z = F.relu(self.bn5(self.tconv1(z)))
        z = self.drop5(z)
        z = F.relu(self.bn6(self.tconv2(z)))
        z = self.drop6(z)
        z = F.relu(self.bn7(self.tconv3(z)))
        z = self.drop7(z)
        z = F.relu(self.tconv4(z))
        return z


def ELBO(mse, mu, logstd):
    kl = -0.5*torch.sum(1+logstd-mu**2-2*torch.exp(logstd))
    return kl+mse


def train(model, train_loader, optimizer, epoch, clip=None):
    model.train()
    total_loss = 0.0
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        counter += 1
        optimizer.zero_grad()
        enc = model.enc(data)
        mu, logstd, z = model.sample_z(enc)
        rec = model.dec(z)
        mse = F.mse_loss(rec, data, reduction='sum')
        loss = ELBO(mse, mu, logstd)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        total_loss += loss.item()/len(data)
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()/len(data)))

    total_loss /= counter
    return total_loss


def val(model, val_loader):
    model.eval()
    test_loss = 0.0
    counter = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            counter += 1
            enc = model.enc(data)
            mu, logstd, z = model.sample_z(enc)
            rec = model.dec(z)
            mse = F.mse_loss(rec, data, reduction='sum')
            test_loss += ELBO(mse, mu, logstd).item()/len(data)
            if counter == len(val_loader.dataset) // len(target):  # last batch of validation loop
                # get reconstruction samples and corresponding original images
                recon_images = rec
                originals = data
        test_loss /= counter
    return test_loss, recon_images, originals


class EarlyStopping:
    # Quick manual implementation of an Early Stopping for use in pytorch. Implementation inspired by
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, wait=5, margin=0., file='saved_model.pt', verbose=False):
        '''
        Standard Early Stopping implementation with basic functionality
        :param wait: int, optional, number of epochs to wait before stopping training early, default is 5
        :param margin: flaot, optional, margin of increase that needs to be reached to be considered an improvement
                       over previous best score, default 0.
        :param file: str, optional, file or file path of the model save, default is current directory and
                     saved_model.pt
        :param verbose: bool, optional, if True prints out increases and saving of model, default False
        '''
        self.wait = wait
        self.margin = margin
        self.file = file
        self.counter = 0
        self.highscore = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss, model):
        curr_score = val_loss

        if self.highscore is None:
            self.highscore = curr_score
            self.save_model(val_loss, model)
        elif self.highscore > curr_score + self.margin:
            self.save_model(val_loss, model)
            self.highscore = curr_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping: {self.counter} out of {self.wait}')
            if self.counter >= self.wait:
                self.early_stop = True

    def save_model(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased from previous best {self.highscore:.5f} to {val_loss:.5f}. '
                  f'Saving model checkpoint')
        torch.save(model.state_dict(), self.file)

