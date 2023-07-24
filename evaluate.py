import torch
from torch import nn, optim
import pickle
import numpy as np
import random
from models import FreqSResolutionTransformer
from utils import load_freq_dataloader
from torch.nn import DataParallel
import os
import re
import random
import time



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
down = 2
root_path = r'preprocessed_data'
# train_experimentIDs = [7]
train_experimentIDs = [7]
test_experimentIDs = [6]
val_experimentIDs = None
indices = False
batch_size = 8

train_loader, _, test_loader, (LR_mean, LR_std), (LR_size, HR_size) = \
    load_freq_dataloader(root_path, train_experimentIDs, test_experimentIDs,
                         val_experimentIDs=val_experimentIDs, indices=indices,
                         down=down, batch_size=batch_size)


val_loader = test_loader
image_size = LR_size
SR_size = HR_size
if down == 2:
    patch_size = (2, 2, 2)
else:
    patch_size = (1, 1, 1)
out_channel = 64
kernel_size = 5
# pre 4
n_conv = 4

# ensure encoder_dim % (scale * patch_size[0])^3 = 0

# pre - 1024
encoder_dim = 1024

# 12
encoder_n_layers = 4
# pre - 8
encoder_heads = 8

encoder_dim_head = encoder_dim // encoder_heads
encoder_mlp_dim = 128

emb_dropout = 0.0
dropout = 0.0
pool = 'cls'
channels = 2
loss_type = 'l1'
lr = 2.5e-4

print('batchsize: ', batch_size, 'patch size', patch_size, 'kernel size: ',
      kernel_size, 'out_channel: ', out_channel, 'learning rate: ', lr,
      'n_conv: ', n_conv, 'n_trans: ', encoder_n_layers)

# --------------------------- finetune part

def load_checkpoint(model, checkpoint_PATH):
    model_CKPT = torch.load(checkpoint_PATH, map_location='cpu')
    model.load_state_dict(model_CKPT['encoder_state_dict'])
    print('loading checkpoint!')
    return model


finetune_model = FreqSResolutionTransformer(image_size=image_size, scale=down, SR_size=SR_size, patch_size=patch_size,
                                            out_channel=out_channel, kernel_size=kernel_size,
                                            encoder_dim=encoder_dim, encoder_n_layers=encoder_n_layers,
                                            encoder_heads=encoder_heads, encoder_dim_head=encoder_dim_head,
                                            encoder_mlp_dim=encoder_mlp_dim,
                                            dropout=dropout, emb_dropout=emb_dropout, masking_ratio=0.,
                                            loss_type=loss_type, pool=pool, channels=channels, HR_channel=2,
                                            n_conv=n_conv)

device = torch.device('cuda:0')
finetune_model = finetune_model.to(device)
# finetune_model = torch.nn.DataParallel(finetune_model.to(device), device_ids=[0, 1])
finetune_optimizer = optim.AdamW(finetune_model.parameters(), lr=lr, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(finetune_optimizer, step_size=50, gamma=0.1)
train_losses = []
val_losses = [0]
val_nrmses = []


checkpoint_path = r'./checkpoints/'
if os.path.exists(checkpoint_path) is not True:
    os.makedirs(checkpoint_path)

if down == 2:
    start_epoch = 60
else:
    start_epoch = 110
    
if start_epoch != 0:
    checkpoint_PATH = checkpoint_path + \
                     'scale_' + str(down) + \
                     '_epoch_' + str(start_epoch) + \
                     '_kernel_' + str(kernel_size) + \
                     '_patch_size_' + str(patch_size[0]) + \
                     'batch_size_' + str(batch_size) + \
                     '_trans_' + str(encoder_n_layers) + \
                     '_finetune.pth.tar'

    model_CKPT = torch.load(checkpoint_PATH,  map_location='cpu')
    finetune_model.load_state_dict(model_CKPT['encoder_state_dict'])
    print('loading checkpoint from: ', checkpoint_PATH)


def predict(model, val_loader, val_nrmses):
    pred_HR_SM, HR_SM = [], []
    with torch.no_grad():
        model.eval()
        for step, (LR_img, HR_img, freqs, coils, _, _) in enumerate(val_loader):
            LR_img = LR_img.to(device)
            freqs, coils = freqs.to(device), coils.to(device)
            pred_HR_img = model.predict(LR_img, freqs, coils, p=1).cpu().numpy()

            pred_HR_SM.append(pred_HR_img)
            HR_SM.append(HR_img.numpy())

            del LR_img, HR_img, freqs, coils

        pred_HR_SM = np.concatenate(pred_HR_SM, 0) * LR_std + LR_mean
        HR_SM = np.concatenate(HR_SM, 0)

        new_pred_HR_SM = np.zeros(pred_HR_SM.shape)
        new_pred_HR_SM[:, :, 2:-1, 2:-1, 2:-1] = pred_HR_SM[:, :, 2:-1, 2:-1, 2:-1]
        pred_HR_SM = new_pred_HR_SM
       
        del new_pred_HR_SM

        comp_HR_SM = HR_SM[:, 0, :, :, :] + 1j * HR_SM[:, 1, :, :, :]
        comp_reco_SM = pred_HR_SM[:, 0, :, :, :] + 1j * pred_HR_SM[:, 1, :, :, :]

        vec_reco_SM = comp_reco_SM.reshape(comp_reco_SM.shape[0], 1, -1)
        vec_HR_SM = comp_HR_SM.reshape(comp_HR_SM.shape[0], 1, -1)

        # alpha = (vec_reco_SM.conj() * vec_HR_SM).sum((1, 2)) / (vec_reco_SM.conj() * vec_reco_SM).sum((1, 2))
        # vec_reco_SM = vec_reco_SM * alpha.reshape(-1, 1, 1)

        reco_SM = vec_reco_SM.reshape(vec_reco_SM.shape[0], 40, 40, 40)
        pickle.dump(reco_SM, open('trans_reco_SM_' + str(down) + '_60.pkl', 'wb'))

        N = vec_reco_SM.shape[-1]
        rmse = np.linalg.norm(vec_reco_SM - vec_HR_SM, 'fro', (1, 2)) / np.sqrt(N)
        val_nrmse = rmse / (np.abs(vec_HR_SM).max((1, 2)) - np.abs(vec_HR_SM).min((1, 2)))
        val_nrmses.append(val_nrmse.mean())

        del pred_HR_img, alpha
        del pred_HR_SM, HR_SM, comp_HR_SM, comp_reco_SM, vec_reco_SM, vec_HR_SM
        # torch.cuda.empty_cache()

        print(start_epoch, 'val nrmse: ', val_nrmses[-1])
        

predict(finetune_model, val_loader, val_nrmses)


