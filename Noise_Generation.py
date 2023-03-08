import Noise_VAE
import torch
import matplotlib.pyplot as plt
from Noise_VAE import train, test
from Data_Setup import construct_dataloaders
import torch.optim as optim

train_loader, val_loader, test_loader = construct_dataloaders('./Data/samples.h5', train_batch_size=64,
                                                              val_batch_size=256, test_batch_size=256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Noise_VAE.ConvVAE().to(device)

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
plt.plot(test_loss, c='orange')
plt.xlabel('epoch')
plt.savefig('./ConvVAE/Loss_plot')