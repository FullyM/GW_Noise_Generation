import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=7, padding=3)  # 128
        self.conv1_2 = nn.Conv2d(6, 12, kernel_size=7, padding=3)  # 128
        self.conv2 = nn.Conv2d(12, 18, kernel_size=5, padding=2)  # 64
        self.conv2_2 = nn.Conv2d(18, 24, kernel_size=5, padding=2)  # 64
        self.conv3 = nn.Conv2d(24, 30, kernel_size=3, padding=1)  # 32
        self.conv3_2 = nn.Conv2d(30, 36, kernel_size=3, padding=1)  # 32
        self.conv4 = nn.Conv2d(36, 42, kernel_size=3, padding=1)  # 16
        self.conv4_2 = nn.Conv2d(42, 48, kernel_size=3, padding=1)  # 16
        self.conv5 = nn.Conv2d(48, 54, kernel_size=3, padding=1)  # 8
        self.conv5_2 = nn.Conv2d(54, 60, kernel_size=3, padding=1)  # 8
        self.bn1 = nn.BatchNorm2d(6)
        self.bn1_2 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(18)
        self.bn2_2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(30)
        self.bn3_2 = nn.BatchNorm2d(36)
        self.bn4 = nn.BatchNorm2d(42)
        self.bn4_2 = nn.BatchNorm2d(48)
        self.bn5 = nn.BatchNorm2d(54)
        self.bn5_2 = nn.BatchNorm2d(60)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.drop3 = nn.Dropout(p=0.1)
        self.drop4 = nn.Dropout(p=0.1)
        self.drop5 = nn.Dropout(p=0.1)

        #self.fc1 = nn.Linear(8*8*60, 1024)
        #self.fc2 = nn.Linear(1024, 512)
        #self.fc3 = nn.Linear(512, 256)
        self.bn_lin1 = nn.BatchNorm1d(1024)
        self.bn_lin2 = nn.BatchNorm1d(512)
        self.bn_lin3 = nn.BatchNorm1d(256)
        self.drop_lin1 = nn.Dropout(p=0.1)
        self.drop_lin2 = nn.Dropout(p=0.1)
        self.drop_lin3 = nn.Dropout(p=0.1)

        self.mu_f = nn.Linear(30, 16)
        self.logstd_f = nn.Linear(30, 16)

        #self.fc1_dec = nn.Linear(16, 60)
        self.bn_declin1 = nn.BatchNorm1d(60)

        self.tconv1 = nn.ConvTranspose2d(16, 54, kernel_size=7, padding=3)  # 8
        self.tconv1_2 = nn.ConvTranspose2d(54, 48, kernel_size=7, padding=3)  # 8
        self.tconv2 = nn.ConvTranspose2d(48, 42, kernel_size=5, padding=2)  # 16
        self.tconv2_2 = nn.ConvTranspose2d(42, 36, kernel_size=5, padding=2)  # 16
        self.tconv3 = nn.ConvTranspose2d(36, 30, kernel_size=3, padding=1)  # 32
        self.tconv3_2 = nn.ConvTranspose2d(30, 24, kernel_size=3, padding=1)  # 32
        self.tconv4 = nn.ConvTranspose2d(24, 18, kernel_size=3, padding=1)  # 64
        self.tconv4_2 = nn.ConvTranspose2d(18, 12, kernel_size=3, padding=1)  # 64
        self.tconv5 = nn.ConvTranspose2d(12, 6, kernel_size=3, padding=1)  # 128
        self.tconv5_2 = nn.ConvTranspose2d(6, 1, kernel_size=3, padding=1)  # 128
        self.tbn1 = nn.BatchNorm2d(54)
        self.tbn1_2 = nn.BatchNorm2d(48)
        self.tbn2 = nn.BatchNorm2d(42)
        self.tbn2_2 = nn.BatchNorm2d(36)
        self.tbn3 = nn.BatchNorm2d(30)
        self.tbn3_2 = nn.BatchNorm2d(24)
        self.tbn4 = nn.BatchNorm2d(18)
        self.tbn4_2 = nn.BatchNorm2d(12)
        self.tbn5 = nn.BatchNorm2d(6)
        self.tdrop1 = nn.Dropout(p=0.1)
        self.tdrop2 = nn.Dropout(p=0.1)
        self.tdrop3 = nn.Dropout(p=0.1)
        self.tdrop4 = nn.Dropout(p=0.1)

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
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.max_pool2d(x, kernel_size=8)
        x = x.flatten(start_dim=1)
        #x = F.relu(self.bn_lin1(self.fc1(x)))
        #x = F.relu(self.bn_lin2(self.fc2(x)))
        #x = self.bn_lin3(self.fc3(x))
        return x

    def dec(self, z):
        #z = F.relu(self.bn_declin1(self.fc1_dec(z)))
        b, l = z.shape
        z = z.view(b, l, 1, 1)
        z = F.interpolate(z, mode='bilinear', scale_factor=8)
        z = F.relu(self.tbn1(self.tconv1(z)))
        z = F.relu(self.tbn1_2(self.tconv1_2(z)))
        z = F.interpolate(z, mode='bilinear', scale_factor=2)
        z = F.relu(self.tbn2(self.tconv2(z)))
        z = F.relu(self.tbn2_2(self.tconv2_2(z)))
        z = F.interpolate(z, mode='bilinear', scale_factor=2)
        z = F.relu(self.tbn3(self.tconv3(z)))
        z = F.relu(self.tbn3_2(self.tconv3_2(z)))
        z = F.interpolate(z, mode='bilinear', scale_factor=2)
        z = F.relu(self.tbn4(self.tconv4(z)))
        z = F.relu(self.tbn4_2(self.tconv4_2(z)))
        z = F.interpolate(z, mode='bilinear', scale_factor=2)
        z = F.relu(self.tbn5(self.tconv5(z)))
        z = self.tconv5_2(z)
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
    def __init__(self, wait=5, margin=0., file='saved_model.pt', start_patience=0, verbose=False, scheduler=None,
                 scheduler_wait=0):
        '''
        Standard Early Stopping implementation with basic functionality. Allows for passing of a learning rate
        scheduler, so that the learning rate is changed first to try and progress training before the training is
        stopped early. Will only do one learning rate step to try and lower the validation loss.
        :param wait: int, optional, number of epochs to wait before stopping training early, default is 5
        :param margin: flaot, optional, margin of increase that needs to be reached to be considered an improvement
                       over previous best score, default 0.
        :param file: str, optional, file or file path of the model save, default is current directory and
                     saved_model.pt
        :param start_patience: int, optional, allows to set an initial number of epochs during which training will
                               not be stopped early
        :param verbose: bool, optional, if True prints out increases and saving of model, default False
        :param scheduler: expects a LRScheduler object, optional, allows to tie in LRS steps with EarlyStopping
        :param scheduler_wait: int, optional, number of epochs without validation loss decrease before making a
                               lr scheduler step, do this step multiple times if wait is a multiple of scheduler wait.
                               Will only do 1 scheduler step so adjust the step size of the scheduler accordingly.
        '''
        self.wait = wait
        self.margin = margin
        self.file = file
        self.counter = 0
        self.highscore = None
        self.early_stop = False
        self.start_patience = start_patience
        self.start_counter = 0
        self.verbose = verbose
        self.scheduler = scheduler
        self.scheduler_wait = scheduler_wait
        self.step = 1

    def __call__(self, val_loss, model):
        curr_score = val_loss

        if self.start_patience > self.start_counter:  # for the first epochs no early stopping will be used
            self.start_counter += 1

        elif self.highscore is None:
            self.highscore = curr_score
            self.save_model(val_loss, model)

        elif self.highscore*(1-self.margin) > curr_score:  # needs to be smaller by relative margin
            self.save_model(val_loss, model)
            self.highscore = curr_score
            self.counter = 0  # reset counter
            self.step = 0  # also reset learning rate step counter

        else:
            self.counter += 1
            self.step += 1
            if self.verbose:
                print(f'Early stopping: {self.counter} out of {self.wait}')
            # will make a lr scheduler step after scheduler wait epochs to improve stagnant training before termination
            if (self.scheduler is not None) and (self.step >= self.scheduler_wait):
                self.scheduler.step()  # make sure the scheduler step size is set appropriately
                self.step = 0  # prevent the scheduler from taking the next step immediately
                if self.verbose:
                    print(f'Adjusting learning rate by factor {self.scheduler.gamma} as validation loss has not'
                          f' improved in the last {self.scheduler_wait} epochs')
            if self.counter >= self.wait:  # if training has not progressed set early stop flag
                self.early_stop = True

    def save_model(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased from previous best {self.highscore:.5f} to {val_loss:.5f}. '
                  f'Saving model checkpoint')
        torch.save(model.state_dict(), self.file)

