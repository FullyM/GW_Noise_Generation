import torch
import torchvision
import glob
import h5py


def convert_png2h5(img_dir, h5_name, im_size):

    img_width = im_size[0]
    img_height = im_size[1]

    h5_file = h5_name+'.h5'

    n_files = len(glob.glob('./'+img_dir+'/*.png'))

    with h5py.File(h5_file, 'w') as h5:
        img_ds = h5.create_dataset('./samples', shape=(n_files, 3, img_width, img_height),
                                   dtype=int)
        label_ds = h5.create_dataset('./labels', shape=(n_files,), dtype=int)
        h5.attrs['length'] = n_files
        for num, file in enumerate(glob.iglob('./'+img_dir+'/*.png')):
            img = torchvision.io.read_image(file, mode=torchvision.io.ImageReadMode.RGB)
            img_ds[num:num+1:, :, :] = img
            label_ds[num] = num


class NoiseDataSet(torch.utils.data.Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.dataset = None
        self.samples_key = 'samples'
        self.labels_key = 'labels'

        with h5py.File(self.path, 'r') as file:
            self.dataset_len = file.attrs('length')

    def __getitem__(self, item):

        if self.dataset is None:
            group = h5py.File(self.path, 'r')
            self.dataset = group.get(self.samples_key)
            self.labels = group.get(self.labels_key)

        image = self.dataset[item]
        label = torchvision.transforms.functional.to_tensor(self.labels[item])

        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.functional.to_tensor(image)

        return image, label

    def __len__(self):
        return self.dataset_len
