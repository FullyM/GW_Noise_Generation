# Script with functions to create GW strain data spectrogram plots
import time
import matplotlib.pyplot as plt


def plot_spectrogram(data, stride, fftlength, overlap=0., vmin=5e-24, vmax=1e-19, draw=False, save=True, name=None, print_file=None, zoom_low=0., zoom_high=0., density=False, q=False, verbose=False):
    '''
    Basic function to plot spectrograms of given GW strain data. Rough implementation, please refer to the variable explanations. Will break if print_file is not specified outside of the function. Will plot the spectrogram
    using the ASD by taking the square root of the spectrogram method output. Old function that is very convoluted that was mostly used for test spectrograms and to test how long a q-transform takes etc.. For generation of
    spectrogram training samples please use q_plot function.
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


def pre_processing(data, min_freq=None, max_freq=None, low_bound=None, high_bound=None, fftlength=None, overlap=0.1):
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

    if min_freq and max_freq:
        data = data.bandpass(min_freq, max_freq)

    elif max_freq:
        data = data.lowpass(max_freq)

    elif min_freq:
        data = data.highpass(min_freq)

    if fftlength:
        data = data.whiten(fftlength, overlap)

    if high_bound:
        data = data.crop(low_bound, high_bound)

    pro_data = data

    return pro_data


def plot_q(data, name, dir_name, q_range=None, whiten=True, f_duration=0.1, show=False, labels=False,
           zoom=None, im_size=(1, 1), dpi=128, **kwargs):
    '''
    Function to calculate the Q-Transform of given gwpy.timeseries.Timeseries object and directly create and save a plot
    of the corresponding spectrogram. Has the option to either produce a mostly normal plot or produces an image of the
    spectrogram without any plot elements i.e. axis, border, labels etc.. Implemented for the purpose of generating
    training samples for generative model training.
    :param data: gwpy.timeseries.Timeseries object, data of which the q-transform should get calculated
    :param name: string, mandatory, name of the file which will contain the plotted spectrogram
    :param dir_name: string, mandatory, name of the output directory the files will be stored in
    :param q_range: tuple of float, optional, range of q's to use for the transform, inverse relation to range of
                    frequencies displayed in the spectrogram, unrelated to f_range
    :param f_range: tuple of floats, optional, range of frequencies to consider for the q-transform, can be specified in
                    kwargs argument
    :param whiten: boolean, optional, if the data should be whitened before the q-transform
    :param f_duration: float, optional, length of the timeseries used to estimate the PSD for whitening
    :param show: boolean, optional, if set to True will show the plotted spectrogram
    :param labels: boolean, optional, if set to True will return only the spectrogram and no plot labels or axis
    :param zoom: tuple of floats, optional, can specify a time range for the spectrogram plot
    :param im_size: tuple of floats, optional, the size of the returned image
    :param dpi: int, optional, the dpi of the saved image
    :return: gwpy.timeseries.Spectrogram object, containing the q-transformed input timeseries object
    '''

    q = data.q_transform(qrange=q_range, whiten=whiten, fduration=f_duration, **kwargs)

    q_plot = q.plot(cmap='viridis', yscale='log')
    fig = plt.figure(q_plot, frameon=labels)  # frameon argument specifies if the figure has a frame or not

    if labels:                      # if one wants a normal plot, not suited for training data
        ax = q_plot.gca()
        if zoom:
            ax.set_xlim(zoom)
        ax.colorbar(label='Strain')  # colorbar label is hardcoded here, rarely used so needs to be manually set here

    else:
        fig.set_size_inches(im_size)
        ax = fig.gca()
        ax.set_axis_off()  # turn of all axis elements
        fig.subplots_adjust(left=0, top=1, bottom=0, right=1)  # removes white borders around the plotted image

    fig.savefig('./'+dir_name+'/'+name, dpi=dpi)  # target directory Q_Plots needs to exist already

    if show:
        fig.show()
        q_plot.show()

    return q
