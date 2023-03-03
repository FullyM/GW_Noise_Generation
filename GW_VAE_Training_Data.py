from gwpy.timeseries import TimeSeries
from Spectrogram_Plots import pre_processing, plot_q
import PIL
import numpy as np

GW_Noise = TimeSeries.fetch_open_data('L1', 'Mar 18 2020 09:00:00', 'Mar 18 2020 09:10:00', cache=True, verbose=True)

GW_Noise_proc = pre_processing(GW_Noise)  # not doing any pre-processing here as qtransform already whitens
GW_Noise_proc = GW_Noise_proc[1000:-1000]

GW_Noise = []
length = len(GW_Noise_proc)//100
for i in range(1, 100):
    GW_Noise.append(GW_Noise_proc[(i-1)*length:i*length])

GW_Noise_q = []
for i in range(30):
    if i % 10 == 0:
        print('Calculating Q-Transform for Data '+str(i))
    GW_Noise_q.append(plot_q(GW_Noise[i], name='Test_'+str(i), dir_name='Q_Plots', q_range=(8, 32), f_duration=2.))

im = []
image_dir = './Q_Plots/'
for i in range(len(GW_Noise_q)):
    im.append(PIL.Image.open(image_dir+'Test_'+str(i)+'.png', 'r'))


pixel_val = []
for i in range(len(im)):
    pixel_val.append(list(im[i].getdata()))


pixel_val = np.array(pixel_val)
pixel_val = pixel_val.transpose((0, 2, 1))
batch, channel, pixels = pixel_val.shape
w = h = int(np.sqrt(pixels))  # should only use quadratic images so the square root of number of pixels is always int
test_pixel = pixel_val.reshape(batch, channel, w, h)
print(test_pixel.shape)
