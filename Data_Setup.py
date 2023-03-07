import torch
import torchvision
import glob
import h5py


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

    img_width = im_size[0]
    img_height = im_size[1]

    h5_file = h5_name+'.h5'

    n_files = len(glob.glob('./'+img_dir+'/*.png'))  # This cycles over all .png files contained in the given directory

    with h5py.File(h5_file, 'w') as h5:
        img_ds = h5.create_dataset('./samples', shape=(n_files, 3, img_width, img_height),  # shape for RGB images
                                   dtype=int)
        label_ds = h5.create_dataset('./labels', shape=(n_files,), dtype=int)  # placeholder labels of image numbers
        h5.attrs['length'] = n_files
        for num, file in enumerate(glob.iglob('./'+img_dir+'/*.png')):
            img = torchvision.io.read_image(file, mode=torchvision.io.ImageReadMode.RGB)  # hardcoded ReadMode
            img_ds[num:num+1:, :, :] = img
            label_ds[num] = num  # change labels here


class NoiseDataSet(torch.utils.data.Dataset):
    '''
    Custom DataSet class to handle the spectrogram images saved in the .h5 file. Very specific to the setup created here
    with hardcoded dataset keys etc. Takes the path of the h5 file and possible transforms. Already returns pytorch
    tensors. Inspired by https://github.com/sara-nl/Packed-Data-Formats/blob/master/datasets.py
    '''
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.dataset = None
        self.samples_key = 'samples'
        self.labels_key = 'labels'

        with h5py.File(self.path, 'r') as file:
            # will this clash with the splitting of the dataset?
            self.dataset_len = file.attrs['length']  # needs the h5 file to have length saved as an attribute

    def __getitem__(self, item):

        if self.dataset is None:
            group = h5py.File(self.path, 'r')
            self.dataset = group.get(self.samples_key)
            self.labels = group.get(self.labels_key)

        image = self.dataset[item]
        label = torch.tensor(self.labels[item])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image)

        return image, label

    def __len__(self):
        return self.dataset_len


def construct_dataloaders(path, train_batch_size, val_batch_size, test_batch_size, transform=None,
                          splits=[0.7, 0.2, 0.1], shuffle=True):
    '''
    Constructs the dataloaders for the custom spectrogram dataset above. Will produce 3 dataloaders for training,
    validation and test samples.
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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader
