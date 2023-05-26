import Noise_VAE
import torch
from Noise_VAE import generate
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs/generated_images')

model = Noise_VAE.ConvVAE()
model.load_state_dict(torch.load('./VAE_iterations/model_checkpoints/saved_model.pt', map_location=torch.device('cpu')))
model.eval()

generations = generate(20, 16, model, gpu=False)
writer.add_images('Generations', generations, 0)
