# Test script for GW noise download and spectrogram production

from gwosc.datasets import event_gps
import matplotlib.pyplot as plt


plt.ion()

gps_time = event_gps('GW150914')
start = int(gps_time) - 600
end = int(gps_time) - 30

from gwpy.timeseries import TimeSeries
data = TimeSeries.fetch_open_data('L1', start, end, verbose=True, cache=True)
print(data.shape)

plot = data.plot()
plot.show()

spectro = data.spectrogram(0.5, fftlength=0.1)**(1/2)

plot_spec = spectro.imshow(norm='log', vmin=5e-24, vmax=1e-19)
ax = plot_spec.gca()
ax.set_yscale('log')
ax.set_ylim(10, 2000)
ax.colorbar(label=r'GW Amplitude')
plot_spec.show()

normal_spectro = spectro.ratio('median')

plot_normal = normal_spectro.imshow(norm='log', vmin=.1, vmax=10, cmap='Spectral_r')
ax_2 = plot_normal.gca()
ax_2.set_yscale('log')
ax_2.set_ylim(10, 2000)
ax_2.colorbar(label=r'Normalised Amplitude')
plot_normal.show()

data_pro = data.highpass(20)
data_pro = data_pro.whiten(4, 2)

spectro2 = data_pro.spectrogram2(fftlength=1/64, overlap=3/256)**(1/2)
print(spectro2.shape)




#plot_spec2 = spectro2.plot(norm='log', yscale='log')
#ax = plot_spec2.gca()
#ax.set_ylim(10, 2000)
#ax.colorbar(label=r'GW Amplitude')
#plot_spec2.show()
