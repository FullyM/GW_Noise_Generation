from gwpy.timeseries import TimeSeries
from Spectrogram_Plots import pre_processing, plot_spectrogram
import numpy as np


GW_Noise = TimeSeries.fetch_open_data('L1', 'Mar 18 2020 09:00:00', 'Mar 18 2020 09:20:00', cache=True, verbose=True)

GW_Noise_proc = GW_Noise  # pre_processing(GW_Noise, min_freq=20)
GW_Noise_proc = GW_Noise_proc[100:-100]

GW_Noise = []
length = len(GW_Noise_proc)//100
for i in range(1, 100):
    GW_Noise.append(GW_Noise_proc[(i-1)*length:i*length])

print(len(GW_Noise))
print(type(GW_Noise[0]))

GW_Noise_q = []
for i in range(10):
    print(GW_Noise[i])
    GW_Noise_q.append(GW_Noise[i].q_transform())

print(len(GW_Noise_q))

