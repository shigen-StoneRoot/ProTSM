import numpy as np
import h5py
import pickle
import re
import os

def get_SM_img(mdf_file, snr_thres=5):
    '''
    Get the SM image data (3D) from the mdf data.
    Real value and Imag value are considered as two channels.
    '''

    # load mdf data
    f = h5py.File(mdf_file, 'r')

    # get background mask
    isBG = f['/measurement/isBackgroundFrame'][:].view(bool)

    # remove background and get SM with shape (C, K, H * W * D), e.g., (3, 3294, 33*33*27)
    # C is the number of receive coils.
    # K is the number of frequency component after FFT
    # H W D are pixel number of 3D System Matrix
    SM = f['/measurement/data'][:, :, :, :].squeeze()[:, :, isBG == False]

    num_K = SM.shape[1]

    # calculate freq
    numFreq = round(f['/acquisition/receiver/numSamplingPoints'][()] / 2) + 1
    rxBandwidth = f['/acquisition/receiver/bandwidth'][()]
    freq = np.arange(0, numFreq) / (numFreq - 1) * rxBandwidth

    if freq.shape[0] != SM.shape[1]:
        assert f['/measurement/isFrequencySelection'][()] == 1
        freq = freq[f['/measurement/frequencySelection'][()]]

    # shape (C, K)
    freq = np.concatenate([freq.reshape(1, -1) for _ in range(SM.shape[0])])

    # shape (C, K)
    coils = np.concatenate([np.array([i] * num_K).reshape(1, -1) for i in range(3)], 0)

    # get low SNR signals mask
    snr = f['calibration']['snr'][:, :, :].squeeze()
    mask = snr > snr_thres
   
    # reserve frequency component with high SNR
    # shape (N, H * W * D)
    # N is the number of reserved frequency component
    high_snr_SM = SM[mask]

    # reserved frequency
    # shape (N, )
    high_snr_freq = freq[mask]

#    pickle.dump(high_snr_freq, open('exp6_freq.pkl', 'wb'))

    high_snr_coil = coils[mask]

    # two channels respectively for Real value and Imag value
    Re_SM, Im_SM = high_snr_SM.real[:, np.newaxis, :], high_snr_SM.imag[:, np.newaxis, :]

    # shape(N, 2, H * W * D)
    SM_img = np.concatenate([Re_SM, Im_SM], 1)

    return SM_img, high_snr_freq, high_snr_coil



SM_size = (37, 37, 37)

mdf_files = [r'raw_mdf_data/6.mdf', r'raw_mdf_data/7.mdf']

snr_thres = 3
for mdf_file in mdf_files:
    print(mdf_file)
    SM_img, freqs, coils = get_SM_img(mdf_file, snr_thres)
    assert SM_img.shape[-1] == SM_size[0] * SM_size[1] * SM_size[2]
    SM_img = SM_img.reshape(SM_img.shape[0], SM_img.shape[1], SM_size[0], SM_size[1], SM_size[2])
    SM_img = np.pad(SM_img, ((0, 0), (0, 0), (2, 1), (2, 1), (2, 1)), 'constant', constant_values=(0, 0))
    experiment_idx = re.findall("\d+", mdf_file)[0]
    experiment_SM_file = r'preprocessed_data/' + experiment_idx + '_SM_freqs.pkl'
    pickle.dump((SM_img, freqs, coils), open(experiment_SM_file, 'wb'))
    print(SM_img.shape, freqs.shape, coils.shape)

