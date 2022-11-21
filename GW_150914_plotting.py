from gwpy.timeseries import TimeSeries
import time
lho = TimeSeries.fetch_open_data('H1', 1126259458, 1126259467, verbose=True, cache=True)
hp = lho.highpass(20)
#hp = lho.lowpass(400)
white = hp.whiten(4, 2).crop(1126259460, 1126259465)
specgram = white.spectrogram2(fftlength=1/16., overlap=15/256.) ** (1/2.)
specgram = specgram.crop_frequencies(20)
plot = specgram.plot(cmap='viridis', yscale='log')
ax = plot.gca()
ax.set_title('LIGO-Hanford strain data around GW150914')
ax.set_xlim(1126259462, 1126259463)
ax.set_ylim(20, 2000)
ax.colorbar(label=r'Strain ASD [1/$\sqrt{\mathrm{Hz}}$]')
plot.show()

t0 = time.perf_counter()

q_traf = lho.q_transform(outseg=(1126259462, 1126259463), highpass=20, whiten=True, tres=0.001)

t1 = time.perf_counter()

q_plot = q_traf.plot(yscale='log')
ax_q = q_plot.gca()
ax_q.colorbar(label='Strain ASD')
q_plot.show()
print(q_traf.shape)



q_time = t1-t0

print(f'The Q-Transform takes {q_time:0.2f} seconds')