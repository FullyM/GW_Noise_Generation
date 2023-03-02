import numpy as np
from gwpy.timeseries import TimeSeries
from gwdatafind import find_urls
from Spectrogram_Plots import plot_q


def sample_processing(detector, start_time, end_time, sample_duration, dir_name, **q_kws):
    '''
    Composite function that downloads data of a specified detector with specified start and end times from GWOSC using
    CVMFS, and then divides it into equal siced chunks of data with duration sample_duration. These data chunks
    are then Q-Transformed using the plot_q function, which also plots the spectrogram, without plot axis or borders
    by default. Used to create and save samples for deep learning training.
    :param detector: string, mandatory, takes either 'H', 'L' or 'V' and refers to the corresponding the detector
    :param start_time: int, mandatory, specifies the start time of the dataset to be downloaded, the specified timeframe
                        between start and end time must have no gaps in the data, otherwise will give back an error
    :param end_time: int, mandatory, specified the end time of the dataset to be downloaded, the specified timeframe
                     between start and end times must have no gaps in the data, otherwise will return an error
    :param sample_duration: float, mandatory, duration of the individual samples to be produced
    :param dir_name: string, mandatory, name of the directory in which the produced spectrograms should be saved,
                     needs to already exist, if not in the current directory must give the path without starting and end
                     '/'
    :param q_kws: keyword arguments to be passed to the plot_q function, for possible arguments look at
                  Spectrogram_Plots.py, where the function is implemented
    :return: Returns nothing, but saves the spectrograms to the specified directory, named 'sample_n' where n is the
             number of the sample
    '''

    file_url1 = find_urls(f'{detector}', f'{detector}1_GWOSC_O3b_4KHZ_R1', start_time, end_time,
                          host='datafind.gw-openscience.org', on_gaps='error')
    time_series1 = TimeSeries.read(file_url1, f'{detector}1:GWOSC-4KHZ_R1_STRAIN', start=start_time, end=end_time)

    samples = []
    sample_length = sample_duration*time_series1.sample_rate.value
    sample_length = int(sample_length)
    sample_number = len(time_series1)//sample_length
    sample_number = int(sample_number)
    for i in range(1, sample_number+1):
        samples.append(time_series1[(i-1)*sample_length:i*sample_length])

    for i in range(sample_number):
        plot_q(samples[i], name='sample_'+str(i), dir_name=dir_name, **q_kws)
