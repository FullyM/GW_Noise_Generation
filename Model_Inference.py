import Noise_VAE
import torch
from Noise_VAE import generate, plot_latent
from torch.utils.tensorboard import SummaryWriter
from Data_Setup import construct_dataloaders
import torchvision

train_loader, val_loader, test_loader = construct_dataloaders('./Data/samples.h5', train_batch_size=64,
                                                              val_batch_size=512, test_batch_size=512,
                                                              transform=torchvision.transforms.ToTensor(),
                                                              num_workers=17, pin_memory=True)

writer = SummaryWriter(log_dir='./runs/generated_images')

model = Noise_VAE.ConvVAE()
model.load_state_dict(torch.load('./VAE_iterations/model_checkpoints/final_model.pt'))
model.eval()

generations = generate(20, 16, model, gpu=False)
writer.add_images('Generations', generations, 0)

latent_visualisation = plot_latent(model, test_loader, 10)
writer.add_figure('Latent Space Plot', latent_visualisation, 0)
