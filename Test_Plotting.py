from gwpy.timeseries import TimeSeries
from Spectrogram_Plots import pre_processing, plot_spectrogram

print_file = open('prints_q.txt', mode='w')     #Needs this file to already exist

GW_200322_1s = TimeSeries.fetch_open_data('L1', 'Mar 22 2020 09:00:00', 'Mar 22 2020 09:00:01', cache=True, verbose=True)

GW_200322_120s = TimeSeries.fetch_open_data('L1', 'Mar 22 2020 09:00:00', 'Mar 22 2020 09:02:00', cache=True, verbose=True)

GW_200322_90m = TimeSeries.fetch_open_data('L1', 'Mar 22 2020 07:30:00', 'Mar 22 2020 09:00:00', cache=True, verbose=True)

GW_200322_Chirp = TimeSeries.fetch_open_data('L1', 'Mar 22 2020 09:00:00', 'Mar 22 2020 09:30:00', cache=True, verbose=True)

GW_200322_Chirp = pre_processing(GW_200322_Chirp, min_freq=20, fftlength=4, overlap=2, low_bound=1268903420, high_bound=1268903520)

GW200224_Chirp = TimeSeries.fetch_open_data('L1', 'Feb 24 2020 22:00:00', 'Feb 24 2020 22:30:00', cache=True, verbose=True)

GW200224_Chirp = pre_processing(GW200224_Chirp, min_freq=20, fftlength=2, overlap=1, low_bound=1266618080, high_bound=1266618180)


Spectrogram_1s = plot_spectrogram(GW_200322_1s, 1/64, 3/256, overlap=2/256, name='GW_200322_1s_', print_file=print_file)

Spectrogram_120s = plot_spectrogram(GW_200322_120s, 0.1, None, overlap=0.01, name='GW_200322_120s_', print_file=print_file)

Spectrogram_90m = plot_spectrogram(GW_200322_90m, 1, 0.5, overlap=0.25, name='GW_200322_90m_', print_file=print_file)

Spectrogram_Chirp = plot_spectrogram(GW_200322_Chirp, 1/32, 1/64, name='GW_200322_Chirp_q_', zoom_low=1268903510.8, zoom_high=1268903511.8, density=True, q=True, print_file=print_file)

Spec_Chirp = plot_spectrogram(GW200224_Chirp, 1/16, 1/32, overlap=7/256, name='GW200224_Chirp_q_', zoom_low=1266618172, zoom_high=1266618173, density=True, q=True, print_file=print_file)

print_file.close()

# TODO add data and construct the spectrogram for a known glitch?
