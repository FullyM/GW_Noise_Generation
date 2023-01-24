import math
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

batch_size_train = 128
batch_size_test = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)),

                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)),
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = list(test_loader)
example_data, example_targets = examples[0]

fig = plt.figure(figsize=(20, 10))
for i in range(40):
    plt.subplot(5,8,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
#plt.savefig('MNIST_print')

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(10, 15, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(15, 20, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(980, 128)
        self.mu_f = nn.Linear(64, 32)
        self.log_std_f = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 128)

        self.tconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.tconv3 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2)
        self.tconv4 = nn.ConvTranspose2d(4, 1, kernel_size=5, padding=2)

    def reparametrization(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + (eps*std)
        return z

    def forward_enc(self, x):
        # encoder
        # first normal CNN architecture with 4 convolutions and 2 max pooling operations, all convolutions get activated by a relu
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))      # fc output to learn features and representation
        return x

    def get_z(self, x):
        # now get mu and log_variance from the fc output
        x = x.view(-1, 2, 64)
        mu = self.mu_f(x[:, 0, :])
        log_std = self.log_std_f(x[:, 1, :])

        # use the reparametrization trick to sample the latent space vector
        z = self.reparametrization(mu, log_std)
        #z = self.fc2(z)     #

        b, len = z.shape
        z = z.view(b, len, 1, 1)
        return mu, log_std, z

    def forward_dec(self, z):
        y = F.relu(self.tconv1(z))
        y = F.relu(self.tconv2(y))
        y = F.relu(self.tconv3(y))
        y = F.relu(self.tconv4(y))
        #y = torch.sigmoid(y)

        return y


def log_normal(z, mu, std):
    p = 1/(z*std*torch.sqrt(2*torch.tensor(math.pi)))*torch.exp(-(torch.log(z-mu)**2)/(2*std**2))
    return p

def final_loss(bce, z, mu, logstd):
    std = torch.exp(logstd)
    z = z.squeeze()
    z = torch.log(z)
    kl = -0.5*torch.sum(1+logstd-mu**2-torch.exp(logstd))
    #kl = torch.sum(log_normal(z, mu, std) - log_normal(z, 0, 1))
    #kl = F.kl_div(z, torch.randn_like(z), reduction='sum')
    return kl+bce


def train(model, train_loader, optimizer, criterion, epoch):
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
        #bce = criterion(rec, data)
        bce = F.mse_loss(rec, data, reduction='sum')
        loss = final_loss(bce, z, mu, logstd)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    total_loss /= counter
    return total_loss


def test(model, test_loader, criterion):
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
            #bce = criterion(rec, data)
            bce = F.mse_loss(rec, data, reduction='sum')
            test_loss += final_loss(bce, z, mu, logstd).item()
            if counter == len(test_loader.dataset) // len(target):
                print(counter)
                recon_images = rec
                originals = data
        test_loss /= counter
    return test_loss, recon_images, originals


model = ConvVAE().to(device)

learning_rate = 0.001
epochs = 30
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

grid_images = []

train_loss = []
test_loss = []

for epoch in range(1, epochs+1):
    print(f'Epoch {epoch} of {epochs}')
    epoch_train_loss = train(model, train_loader, optimizer, criterion, epoch)
    epoch_test_loss, recon_images, original_images = test(model, test_loader, criterion)
    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)


fig = plt.figure(figsize=(20, 10))
for i in range(40):
  plt.subplot(5,8,i+1)
  plt.tight_layout()
  recon_images = recon_images.cpu()
  plt.imshow(recon_images[i][0], cmap='gray', interpolation='none')
  #plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.savefig('MNIST_recon')


fig = plt.figure(figsize=(20, 10))
for i in range(40):
  plt.subplot(5,8,i+1)
  plt.tight_layout()
  original_images = original_images.cpu()
  plt.imshow(original_images[i][0], cmap='gray', interpolation='none')
  #plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.savefig('MNIST_originals')

fig1 = plt.figure()
plt.plot(train_loss)
plt.plot(test_loss, c='orange')
plt.xlabel('epoch')
plt.savefig('Loss_plot')

def gen_new(D=32, N=40):
    z = np.random.multivariate_normal(np.zeros(D), np.diag(np.ones(D)), N)
    z = torch.tensor(z, dtype=torch.float).to(device)
    n, d = z.shape
    z = z.view(n, d, 1, 1)
    gen_images = model.forward_dec(z)
    return gen_images


generated_images = gen_new()
generated_images = generated_images.cpu()
generated_images = generated_images.detach()

fig = plt.figure(figsize=(20, 10))
for i in range(40):
  plt.subplot(5,8,i+1)
  plt.tight_layout()

  plt.imshow(generated_images[i][0], cmap='gray', interpolation='none')
  #plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.savefig('MNIST_generations')
