from gwpy.timeseries import TimeSeries
from Spectrogram_Plots import pre_processing, plot_spectrogram
import PIL
import numpy as np
import matplotlib.pyplot as plt

GW_Noise = TimeSeries.fetch_open_data('L1', 'Mar 18 2020 09:00:00', 'Mar 18 2020 09:10:00', cache=True, verbose=True)

GW_Noise_proc = pre_processing(GW_Noise)#, fftlength=0.5, overlap=0.1)
GW_Noise_proc = GW_Noise_proc[1000:-1000]

GW_Noise = []
length = len(GW_Noise_proc)//100
for i in range(1, 100):
    GW_Noise.append(GW_Noise_proc[(i-1)*length:i*length])

print(len(GW_Noise))
print(type(GW_Noise[0]))

GW_Noise_q = []
for i in range(10):
    plot_spectrogram(GW_Noise[i], stride=0, fftlength=0, name='Test_'+str(i)+'_', q=True, density=True)

im = []
image_dir = './Q_Plots/'
for i in range(10):
    im.append(PIL.Image.open(image_dir+'Test_'+str(i)+'_Spectrogram.png', 'r'))

print(len(im))

pixel_val = []
for i in range(10):
    print('Getting Pixel values for Spectrogram '+str(i))
    pixel_val.append(list(im[i].getdata()))

print(pixel_val[0][0])
print(len(pixel_val[0]))

resized_ims = []
for i in range(10):
    resized_ims.append(im[i].resize((128, 128), resample=PIL.Image.Resampling.BOX))

pixel_val_resized = []
for i in range(10):
    print('Getting Pixel values for resized Spectrogram ' + str(i))
    pixel_val_resized.append(list(resized_ims[i].getdata()))

print(pixel_val_resized[0][0])
print(len(pixel_val_resized[0]))
pixel_val_resized = np.array(pixel_val_resized)
print(pixel_val_resized.shape)
test_pixel = pixel_val_resized.reshape(10, 4, 128, 128)
print(test_pixel.shape)
print(test_pixel[0])
