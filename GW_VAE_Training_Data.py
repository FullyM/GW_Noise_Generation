from gwpy.timeseries import TimeSeries
from Spectrogram_Plots import pre_processing, plot_spectrogram
import numpy as np
import PIL


GW_Noise = TimeSeries.fetch_open_data('L1', 'Mar 18 2020 09:00:00', 'Mar 18 2020 09:10:00', cache=True, verbose=True)

GW_Noise_proc = pre_processing(GW_Noise, min_freq=20, max_freq=1000, fftlength=4, overlap=1)
GW_Noise_proc = GW_Noise_proc[100:-100]

GW_Noise = []
length = len(GW_Noise_proc)//100
for i in range(1, 100):
    GW_Noise.append(GW_Noise_proc[(i-1)*length:i*length])

print(len(GW_Noise))
print(type(GW_Noise[0]))

GW_Noise_q = []
for i in range(10):
    plot_spectrogram(GW_Noise[i], stride=0, fftlength=0, name='Test_'+str(i)+'_', q=True)

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
