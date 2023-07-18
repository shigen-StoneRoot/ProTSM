import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import random
import monai
from monai.transforms import RandSpatialCrop
from torch.nn.functional import interpolate

transform = torch.from_numpy

class FreqSystemMatrixDataset(Dataset):
    def __init__(self, HR_SM, freqs, coils, transform=None, down=2, mode='train', log=True, LR_mean=None, LR_std=None):

        self.mode = mode
        assert self.mode in ['train', 'val', 'test']

        # (n_sample, channel, H, W, D)
        self.HR_SM = HR_SM


        # (n_sample, )
        self.freqs = freqs.reshape(-1, 1)
        if log:
            self.freqs = np.log2(self.freqs + 1)

        # self.freqs = self.freqs / self.freqs.max()
        if self.mode == 'train':
            self.freqs = (self.freqs - np.mean(self.freqs)) / np.std(self.freqs)
        else:
            self.freqs = (self.freqs - 17.725051363430566) / 0.7258868240597393 # openMPI data
        self.coils = coils.reshape(-1, 1)
            
        self.HR_SM = self.HR_SM.transpose(0, 1, 4, 2, 3)
        self.HR_shp = (self.HR_SM.shape[2], self.HR_SM.shape[3], self.HR_SM.shape[4])

        self.transform = transform
        self.down = down
        assert down in [2, 3, 4, 5]

        self.LR_SM = self.down_sampling()

        # store image size
        self.HR_size, self.LR_size = self.HR_SM.shape[2:], self.LR_SM.shape[2:]

        # # normalize
        if LR_mean is None:
            if down == 2:
                LR_mean = np.mean(self.LR_SM, (2, 3, 4)).reshape(self.LR_SM.shape[0], self.LR_SM.shape[1], 1, 1, 1)
                LR_std = np.std(self.LR_SM, (2, 3, 4)).reshape(self.LR_SM.shape[0], self.LR_SM.shape[1], 1, 1, 1)

            else:
                LR_mean = np.mean(self.LR_SM, (1, 2, 3, 4)).reshape(self.LR_SM.shape[0], 1, 1, 1, 1)
                LR_std = np.std(self.LR_SM, (1, 2, 3, 4)).reshape(self.LR_SM.shape[0], 1, 1, 1, 1)



        self.LR_mean, self.LR_std = LR_mean, LR_std
        self.LR_SM = (self.LR_SM - LR_mean) / LR_std

        #
        # for val and test mode, HR SM are not normalized
        if self.mode == 'train':
            self.HR_SM = (self.HR_SM - LR_mean) / LR_std

        # self.HR_SM = (self.HR_SM - LR_mean) / LR_std

    def __len__(self):
        self.length = self.HR_SM.shape[0]
        return self.length

    def __getitem__(self, idx):
        HR_img, LR_img = self.HR_SM[idx], self.LR_SM[idx]

        freq = self.freqs[idx]
        coil = self.coils[idx]

        HR_img = self.transform(HR_img).float()
        LR_img = self.transform(LR_img).float()


        freq = self.transform(freq).float()
        coil = self.transform(coil).long()

        if self.mode in ['test']:
            LR_mean, LR_std = self.LR_mean[idx], self.LR_std[idx]

            return LR_img, HR_img, freq, coil, LR_mean, LR_std

        else:

            return LR_img, HR_img, freq, coil


    def get_img_size(self):
        return self.LR_size, self.HR_size

    def get_LR_mean_and_std(self):
        return self.LR_mean, self.LR_std


    def down_sampling(self):
        sampling_indices = []
        for axis in range(3):
            indices = np.arange(self.HR_shp[axis])
            if self.HR_shp[axis] % self.down != 0:
                integrat_length = self.HR_shp[axis] - self.HR_shp[axis] % self.down
                rest_mid = int((self.HR_shp[axis] - integrat_length) / 2 + 0.5) + integrat_length - 1
                grid_indices = indices[: integrat_length][1::self.down]
                rest_ind = indices[rest_mid]
                down_sampling_indices = np.concatenate([grid_indices, [rest_ind]])
            else:
                down_sampling_indices = indices[1::self.down]
            sampling_indices.append(down_sampling_indices)

        LR_SM = self.HR_SM[:, :, sampling_indices[0]] \
                          [:, :, :, sampling_indices[1]] \
                          [:, :, :, :, sampling_indices[2]]

        return LR_SM


def load_freq_dataset(root_path, experimentIDs, down=2, transform=None, mode='train', idx=None, LR_mean=None, LR_std=None):
    all_SMs = []
    all_freqs = []
    all_coils = []
    for exp_id in experimentIDs:
        cur_exp_SM_path = os.path.join(root_path, str(exp_id) + '_SM_freqs') + '.pkl'
        cur_SM, freqs, coils = pickle.load(open(cur_exp_SM_path, 'rb'))
        all_SMs.append(cur_SM)
        all_freqs.append(freqs)
        all_coils.append(coils)

    all_SMs = np.concatenate(all_SMs, 0)
    all_freqs = np.concatenate(all_freqs, 0)
    all_coils = np.concatenate(all_coils, 0)
    if idx is not None:
        all_SMs = all_SMs[idx]
        all_freqs = all_freqs[idx]
        all_coils = all_coils[idx]
    print(all_SMs.shape, all_freqs.shape)
    dataset = FreqSystemMatrixDataset(all_SMs, all_freqs, all_coils, transform, down, mode=mode, LR_mean=LR_mean, LR_std=LR_std)
    return dataset


def load_freq_dataloader(root_path, train_experimentIDs, test_experimentIDs, val_experimentIDs=None,
                    indices=False, batch_size=16, down=2):

    train_dataset = load_freq_dataset(root_path, train_experimentIDs, down, transform=transform)
    LR_size, HR_size = train_dataset.get_img_size()
    test_dataset = load_freq_dataset(root_path, test_experimentIDs, down, transform=transform,
                                     mode='test')

    LR_mean, LR_std = test_dataset.get_LR_mean_and_std()

    # loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    if down == 3:
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    else:
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    val_loader = None

    return train_loader, val_loader, test_loader, (LR_mean, LR_std), (LR_size, HR_size)






