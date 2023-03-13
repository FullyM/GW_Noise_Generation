import torchvision.transforms

import Noise_VAE
import torch
import matplotlib.pyplot as plt
from Noise_VAE import train, val
from Data_Setup import construct_dataloaders
import torch.optim as optim

# Need to use ToTensor here in order to reshape the images into the correct shape and scale to [0, 1.] as this makes
# training easier and torch by default only accepts floats as input
train_loader, val_loader, test_loader = construct_dataloaders('./Data/samples.h5', train_batch_size=64,
                                                              val_batch_size=256, test_batch_size=256,
                                                              transform=torchvision.transforms.ToTensor())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Noise_VAE.ConvVAE().to(device)

learning_rate = 0.001
epochs = 50
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

grid_images = []

train_loss = []
val_loss = []

for epoch in range(1, epochs+1):
    print(f'Epoch {epoch} of {epochs}')
    epoch_train_loss = train(model, train_loader, optimizer, epoch)
    epoch_val_loss, recon_images, original_images = val(model, val_loader)
    train_loss.append(epoch_train_loss)
    val_loss.append(epoch_val_loss)


fig = plt.figure(figsize=(20, 10))
recon_images = recon_images.cpu().numpy().transpose((0, 2, 3, 1))
for i in range(40):
  plt.subplot(5,8,i+1)
  plt.tight_layout()
  plt.imshow(recon_images[i], interpolation='none')
  plt.xticks([])
  plt.yticks([])
#plt.savefig('./ConvVAE/MNIST_recon')


fig = plt.figure(figsize=(20, 10))
original_images = original_images.cpu().numpy().transpose((0, 2, 3, 1))
for i in range(40):
    plt.subplot(5,8,i+1)
    plt.tight_layout()
    plt.imshow(original_images[i], interpolation='none')
    # plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
#plt.savefig('./ConvVAE/MNIST_originals')

fig1 = plt.figure()
plt.plot(train_loss)
plt.plot(val_loss, c='orange')
plt.xlabel('epoch')
#plt.savefig('./ConvVAE/Loss_plot')