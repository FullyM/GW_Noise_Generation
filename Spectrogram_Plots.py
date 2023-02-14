# Script with functions to create GW strain data spectrogram plots
import time
import matplotlib.pyplot as plt


def plot_spectrogram(data, stride, fftlength, overlap=0., vmin=5e-24, vmax=1e-19, draw=False, save=True, name=None, print_file=None, zoom_low=0., zoom_high=0., density=False, q=False, verbose=False):
    '''
    Basic function to plot spectrograms of given GW strain data. Rough implementation, please refer to the variable explanations. Will break if print_file is not specified outside of the function. Will plot the spectrogram
    using the ASD by taking the square root of the spectrogram method output.
    :param data: gwpy.Timeseries timeseries data
    :param stride: time domain bin width of the spectrogram
    :param fftlength: frequeny domain bin width of the spectrogram
    :param overlap: optional, how much the different ffts in the time bins are supposed to overlap
    :param vmin: optional, min frequency of the colorbar
    :param vmax: optional, max frequency of the colorbar
    :param draw: optional, defaults to False, will draw the spectrogram if set to True
    :param save: optional, defaults to True, will save the spectrogram plot with the given name with the added string 'spectrogram' at the end, will also save a plot of the time series data input with 'Timeseries' added to name
    :param name: optional, only necessary when save is set to True, the name under which the spectrogram plot and time domain plot are to be saved
    :param timer: optional, defaults to True, will time how long it takes to calculate the spectrogram and return a print output
    :param print_file: optional, set to print_file, will save the print statements to the print_file variable that needs to be specified outside of the function with 'open('file.txt', mode='w')', rough implementation
    :param zoom_low: optional, lower bound for the x axis of the spectrogram plot, defaults to 0
    :param zoom_high: optional, upper bound for the x axis of the spectrogram plot, defaults to 0 and the zoom will only happen if zoom_high is not set to this default
    :return: returns the calculated spectrogram for the given input data and settings
    '''

    t0 = time.perf_counter()
    if not q:
        spectrogram = data.spectrogram(stride, fftlength, overlap=overlap, window='tukey')**(1/2)       # taking the square root to work with the ASD
        # spectrogram = spectrogram.crop_frequencies(20)
    if q:
        spectrogram = data.q_transform(qrange=(8, 32))
    t1 = time.perf_counter()
    spec_calc_time = t1-t0

    if verbose:
        print(f'The Spectrogram has dimensions {spectrogram.shape} for a time series of length {data.duration:0.0f} with stride {stride:0.6f} and fftlength', file=print_file) # print_file needs to exist, otherwise remove this

    duration = data.duration

    if verbose:
        print(f'The Spectrogram calculation took {spec_calc_time:0.2f} seconds for {duration:0.0f} of data', file=print_file)   # print_file needs to exist, remove this if print statemens should not be logged in a .txt file

    if not density:
        spec_plot = spectrogram.imshow(vmin=vmin, vmax=vmax)
        ax = spec_plot.gca()
        ax.set_yscale('log')
        if zoom_high != 0:                                      # with current implementation can only zoom when zoom_high is not 0, can not individually change the lower bound
            ax.set_xlim(zoom_low, zoom_high)
        # hardcoded label of the colourbar, need to change here
        ax.colorbar(label=r'GW Amplitude')

    else:
        spec_plot = spectrogram.plot(cmap='viridis', yscale='log')
        fig = plt.figure(spec_plot, frameon=False)
        fig.set_size_inches(1, 1)
        ax = fig.gca()
        ax.set_axis_off()
        fig.subplots_adjust(left=0, top=1, bottom=0, right=1)
        if zoom_high != 0:  # with current implementation can only zoom when zoom_high is not 0, can not individually change the lower bound
            ax.set_xlim(zoom_low, zoom_high)
        # hardcoded label of the colourbar, need to change here
        #ax.colorbar(label=r'Strain ASD')

    time_plot = data.plot()
    ax = time_plot.gca()
    if zoom_high != 0:
        ax.set_xlim(zoom_low, zoom_high)

    if draw:
        spec_plot.show()
    if save:
        fig.savefig('./Q_Plots/'+name+'Spectrogram', dpi=128)
        time_plot.savefig('./Q_Plots/'+name+'Timeseries')

    return spectrogram


def pre_processing(data, min_freq=0., max_freq=0., low_bound=0., high_bound=0., fftlength=None, overlap=0.):
    '''
    Function which bundles various pre-processing methods for gwpy.Timeseries timeseries objects for pre-processing
    :param data:
    :param min_freq:
    :param max_freq:
    :param low_bound:
    :param high_bound:
    :param fftlength:
    :param overlap:
    :return:
    '''

    if min_freq != 0 and max_freq != 0:
        data = data.bandpass(min_freq, max_freq)

    elif max_freq != 0:
        data = data.lowpass(max_freq)

    elif min_freq != 0:
        data = data.highpass(min_freq)

    if fftlength is not None:
        data = data.whiten(fftlength, overlap)

    if high_bound != 0:
        data = data.crop(low_bound, high_bound)

    pro_data = data

    return pro_data

