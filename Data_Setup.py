import matplotlib.pyplot as plt
import gc
from gwpy.timeseries import TimeSeries
from gwdatafind import find_urls
import torch
import torchvision
import glob
import h5py
import numpy as np


def plot_q(data, name, dir_name, q_range=(4, 64), whiten=True, f_duration=0.1, show=False, labels=False,
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
    # this might be slightly illegal, forcing gwpy.plot object into matplotlib figure, might be the cause for the
    # memory leak mentioned below, though I don't know of a better way to do it, it works for now
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
        plt.close('all')  # need to close all figures to prevent memory bloat

    gc.collect()  # for some reason closing the figures is not enough, need to manually garbage collect
    return q


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


def convert_png2h5(img_dir, h5_name, im_size):
    '''
    Converts png files located in img_dir into one .h5 file. The file is a group with asssociated attribute
    length which is the total number of files read in, consisting of two datasets called 'samples' and 'labels'
    which contain the images as pixel information and the image's number respectively. Images need to be RGB images
    :param img_dir: string, path to the directory containing the images, starting from the current directory
    :param h5_name: string, name of the h5 file to be constructed
    :param im_size: tuple of ints, size of the images in pixels, needs to be quadratic and the same across all images,
                    of the form width, height
    :return: returns nothing, produces a h5 file saved in the current directory
    '''

    img_height = im_size[0]
    img_width = im_size[1]

    h5_file = h5_name+'.h5'

    n_files = len(glob.glob('./'+img_dir+'/*.png'))  # This scans ALL .png files contained in the given directory

    with h5py.File(h5_file, 'w') as h5:
        img_ds = h5.create_dataset('./samples', shape=(n_files, img_height, img_width, 3),  # shape for RGB images
                                   dtype=np.uint8)  # uint8 better interacts with torchvision to_tensor
        label_ds = h5.create_dataset('./labels', shape=(n_files,), dtype=int)  # placeholder labels of image numbers
        h5.attrs['length'] = n_files
        for num, file in enumerate(glob.iglob('./'+img_dir+'/*.png')):
            image = torchvision.io.read_image(file, mode=torchvision.io.ImageReadMode.RGB)  # hardcoded ReadMode
            # torchvision read_image reads in the image as CxHxW but generally images should be read in as HxWxC
            # this unnecessary transposing is O(1) complexity so no computational cost
            image = image.transpose(1, 0)
            image = image.transpose(1, 2)
            img_ds[num:num+1:, :, :] = image
            label_ds[num] = num  # change labels here


class NoiseDataSet(torch.utils.data.Dataset):
    '''
    Custom DataSet class to handle the spectrogram images saved in the .h5 file. Very specific to the setup created here
    with hardcoded dataset keys etc. Takes the path of the h5 file and possible transforms. Already returns pytorch
    tensors. Inspired by https://github.com/sara-nl/Packed-Data-Formats/blob/master/datasets.py
    '''
    def __init__(self, path, transform):
        # transform should almost always be at least torchvision.transforms.ToTensor(), set to None for no transform
        self.path = path
        self.transform = transform

        self.dataset = None
        self.samples_key = 'samples'
        self.labels_key = 'labels'

        with h5py.File(self.path, 'r') as file:
            self.dataset_len = file.attrs['length']  # needs the h5 file to have length saved as an attribute

    def __getitem__(self, item):

        if self.dataset is None:
            group = h5py.File(self.path, 'r')
            self.dataset = group.get(self.samples_key)
            self.labels = group.get(self.labels_key)

        image = self.dataset[item]
        label = torch.tensor(self.labels[item])

        if self.transform:
            # defaults to torchvision.transforms.ToTensor for transposing and rescaling to [0, 1.], needs to be kept
            image = self.transform(image)
        else:
            # to at least ensure expected dimensions of CxHxW
            image = image.transpose((2, 0, 1))
            # this will return int tensors with values [0, 255]
            image = torch.tensor(image)
        return image, label

    def __len__(self):
        return self.dataset_len


def construct_dataloaders(path, train_batch_size, val_batch_size, test_batch_size, transform=None,
                          splits=[0.7, 0.2, 0.1], shuffle=True, **kwargs):
    '''
    Constructs the dataloaders for the custom spectrogram dataset above. Will produce 3 dataloaders for training,
    validation and test samples. This expects the .h5 file with the data to already exist. If the data does not exist
    yet, use convert_png2h5 first. Accepts keyword arguments for the DataLoader class.
    :param path: string, path to the h5 file containing the samples, relative to the current directory
    :param train_batch_size: int, batch size for training samples
    :param val_batch_size: int, batch size for validation samples
    :param test_batch_size: int, batch size for test samples
    :param transform: optional, torchvision transform, possible transformation of training data to be performed
    :param splits: optional, list of floats, giving the fraction of training, validation and test data
    :param shuffle: optional, boolean, specifies if the data should be shuffled after each epoch, defaults to true
    :return: 3 dataloader objects for train, validation and test data
    '''

    data = NoiseDataSet(path, transform=transform)
    train_data, val_data, test_data = torch.utils.data.random_split(data, splits)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=shuffle, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=shuffle, **kwargs)

    return train_loader, val_loader, test_loader
