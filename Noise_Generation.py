import torchvision.transforms
import Noise_VAE
import torch
from Noise_VAE import train, val, EarlyStopping
from Data_Setup import construct_dataloaders
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Need to use ToTensor here in order to reshape the images into the correct shape and scale to [0, 1.] as this makes
# training easier and torch by default only accepts floats as input
# setting num_workers to 17 here as I usually work with 18 CPUs on Snellius
train_loader, val_loader, test_loader = construct_dataloaders('./Data/samples.h5', train_batch_size=64,
                                                              val_batch_size=512, test_batch_size=512,
                                                              transform=torchvision.transforms.ToTensor(),
                                                              num_workers=17, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='./runs/small_kernel')

model = Noise_VAE.ConvVAE().to(device)

learning_rate = 0.01
epochs = 500
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5, verbose=True)
scheduler2 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.001, total_iters=10, verbose=True)
scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5],
                                            verbose=True)
scheduler_ES = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, verbose=True)
wait = 5
scheduler_wait = 2
stop = EarlyStopping(wait=wait, margin=0.01, file='./model_checkpoints/saved_model.pt', verbose=True,
                     start_patience=50)#, scheduler=scheduler_ES, scheduler_wait=scheduler_wait)

for epoch in range(1, epochs+1):
    print(f'Epoch {epoch} of {epochs}')
    epoch_train_loss = train(model, train_loader, optimizer, epoch, clip=None)
    epoch_val_loss, recon_images, original_images = val(model, val_loader)
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
    writer.add_images('Originals', original_images.cpu()[:10], epoch)
    writer.add_images('Reconstructions', recon_images.cpu()[:10], epoch)
    total = 0
    num_par = 0
    for n, par in model.named_parameters():
        if ('bias' not in n) and ('bn' not in n):  # batch norm parameters are not too important for model performance
            # print(f'Gradients of parameter {n} have norm {par.grad.norm(2)}, mean {par.grad.abs().mean()}'
                    # f' and max {par.grad.abs().max()}')
            num_par += 1
            # TODO add individual layer parameters instead of mean over model for better overview
            # Looking at gradient mean without considering bias to judge if gradients explode or die out
            total += par.grad.abs().mean().item()
    writer.add_scalar('Gradient Norm', total/num_par, epoch)
    stop(epoch_val_loss, model)
    if stop.early_stop:
        print(f'Training stopped early because Validation loss has not decreased in the past {wait} epochs')
        print(f'Best validation loss achieved during training: {stop.highscore:.4f}, in epoch {epoch-wait}')
        break
    #scheduler.step()

