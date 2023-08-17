import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchinfo import summary

batch_size_train = 128
batch_size_test = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        # self.conv4 = nn.Conv2d(15, 20, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(980, 128)
        #self.fc2 = nn.Linear(256, 128)
        self.mu_f = nn.Linear(64, 2)
        self.log_std_f = nn.Linear(64, 2)
        self.fc3 = nn.Linear(2, 32)

        self.tconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5)
        self.tconv2 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.tconv3 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2)
        self.tconv4 = nn.ConvTranspose2d(4, 1, kernel_size=5, padding=2)

    def reparametrization(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(log_std)
        z = mu + (eps*std)
        return z

    def forward_enc(self, x):
        # encoder
        # first normal CNN architecture with 2 convolutions and 2 max pooling operations
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)  # fc output to learn features and representation
        #x = self.fc2(x)
        return x

    def get_z(self, x):
        # now get mu and log_variance from the fc output
        b, l = x.shape
        x = x.view(b, 2, l//2)
        mu = self.mu_f(x[:, 0, :])
        log_std = self.log_std_f(x[:, 1, :])

        # use the reparametrization trick to sample the latent space vector
        z = self.reparametrization(mu, log_std)

        return mu, log_std, z

    def forward_dec(self, z):
        z = self.fc3(z)
        b, l = z.shape
        z = z.view(b, l, 1, 1)
        y = F.relu(self.tconv1(z))
        y = F.relu(self.tconv2(y))
        y = F.relu(self.tconv3(y))
        y = F.relu(self.tconv4(y))

        return y

    def forward(self, x):
        x = self.forward_enc(x)
        mu, logstd, z = self.get_z(x)
        x = self.forward_dec(z)


class LinVAE(nn.Module):

    def __init__(self):
        super(LinVAE, self).__init__()
        self.lin1 = nn.Linear(28*28, 512)
        self.lin2 = nn.Linear(512, 128)
        #self.lin3 = nn.Linear(256, 128)
        self.lin_mu = nn.Linear(64, 2)
        self.lin_logstd = nn.Linear(64, 2)

        self.tlin1 = nn.Linear(2, 512)
        self.tlin2 = nn.Linear(512, 28*28)

    def forward_enc(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def get_z(self, x):
        b, l = x.shape
        x = x.view(b, 2, l//2)
        mu = self.lin_mu(x[:, 0, :])
        logstd = self.lin_logstd(x[:, 1, :])
        z = self.reparametrization(mu, logstd)
        return mu, logstd, z

    def reparametrization(self, mu, logstd):
        eps = torch.randn_like(logstd)
        std = torch.exp(logstd)
        z = mu + std*eps
        return z

    def forward_dec(self, z):
        y = F.relu(self.tlin1(z))
        y = self.tlin2(y)
        b, _ = y.shape
        y = y.view(b, 1, 28, 28)
        return y


def final_loss(mse, mu, logstd):
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
        loss = final_loss(mse, mu, logstd)
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
            test_loss += final_loss(mse, mu, logstd).item()/len(data)
            if counter == len(test_loader.dataset) // len(target):
                recon_images = rec
                originals = data
        test_loss /= counter
    return test_loss, recon_images, originals


model = ConvVAE().to(device)


learning_rate = 0.001
epochs = 50
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

grid_images = []

train_loss = []
test_loss = []

for epoch in range(1, epochs+1):
    print(f'Epoch {epoch} of {epochs}')
    epoch_train_loss = train(model, train_loader, optimizer, epoch)
    epoch_test_loss, recon_images, original_images = test(model, test_loader)
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
plt.savefig('./ConvVAE/MNIST_recon')


fig = plt.figure(figsize=(20, 10))
for i in range(40):
    plt.subplot(5,8,i+1)
    plt.tight_layout()
    original_images = original_images.cpu()
    plt.imshow(original_images[i][0], cmap='gray', interpolation='none')
    # plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('./ConvVAE/MNIST_originals')

fig1 = plt.figure()
plt.plot(train_loss)
plt.plot(test_loss, c='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Validation Loss'])
plt.savefig('./ConvVAE/Loss_MNIST')

def gen_new(D=32, N=40):
    z = np.random.multivariate_normal(np.zeros(D), np.diag(np.ones(D)), N)
    z = torch.tensor(z, dtype=torch.float).to(device)
    n, d = z.shape
    #z = z.view(n, d, 1, 1)
    gen_images = model.forward_dec(z)
    return gen_images


generated_images = gen_new(D=2)
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
plt.savefig('./ConvVAE/MNIST_generations')


def plot_latent(model, data_loader, num_batches=100, folder='./ConvVAE/'):
    fig_latent = plt.figure()
    for i, (data, target) in enumerate(data_loader):
        x = model.forward_enc(data.to(device))
        _, _, z = model.get_z(x)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=target, cmap='tab10')
        if i > num_batches:
            break
    plt.colorbar()
    plt.savefig(folder+'latent_MNIST')


plot_latent(model, test_loader)

summary(model, input_size=(batch_size_train, 1, 28, 28))

