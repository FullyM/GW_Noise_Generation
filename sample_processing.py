import numpy as np
from gwpy.timeseries import TimeSeries
from gwdatafind import find_urls
from Spectrogram_Plots import plot_q

def sample_processing(detector, start_time, end_time, sample_duration, **q_kws):

    file_url1 = find_urls('H', 'H1_GWOSC_O3b_4KHZ_R1', start_time, end_time, host='datafind.gw-openscience.org',
                          on_gaps='error')
    time_series1 = TimeSeries.read(file_url1, 'H1:GWOSC-4KHZ_R1_STRAIN', start=start_time, end=end_time)

    samples = np.array([])
    sample_length = sample_duration*time_series1.sample_rate.value
    sample_number = len(time_series1)//sample_length
    for i in range(1, sample_number+1):
        np.append(samples, time_series1[(i-1)*sample_length:i*sample_length])

    spectrograms = np.array([])
    for i in range(sample_number):
        np.append(spectrograms, plot_q(samples[i], name='sample_'+str(i), q_range=q_kws.pop('q_range'),
                                       f_range=q_kws.pop('f_range'), whiten=q_kws.pop('whiten'),
                                       f_duration=q_kws.pop('f_duration'), im_size=q_kws.pop('im_size'),
                                       dpi=q_kws.pop('dpi')))

    for i in range(sample_number):
        plot_q(spectrograms[i])



