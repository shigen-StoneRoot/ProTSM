import torch
from torch import nn, optim
import pickle
import numpy as np
import random
# from models import SResolutionPixelTransformer, SemiSResolutionTransformer, FreqSResolutionTransformer
from models import FreqSResolutionTransformer
from utils import load_freq_dataloader
from torch.nn import DataParallel
import os
import re
import random
import time
from torch.distributions import Beta
import math


# old least 0.1
def adjust_learning_rate(optimizer, init_lr, epoch, warmup_epochs, total_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        r = epoch / warmup_epochs
    else:
        r = 1.0
        # r = 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * r
        return None


seed = 123
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


if down == 2:
    total_epochs = 60
else:
    total_epochs = 110

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
encoder_layers = list(map(id, finetune_model.encoder.parameters()))
others_layers = list(filter(lambda p: id(p) not in encoder_layers, finetune_model.parameters()))

params = [
    {"params": finetune_model.encoder.parameters(), "lr": lr * 0.5},
    {"params": others_layers},

    ]

finetune_optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5, betas=(0.9, 0.99))

scheduler = torch.optim.lr_scheduler.StepLR(finetune_optimizer, step_size=50, gamma=0.1)
train_losses = []
val_losses = [0]
val_nrmses = []


checkpoint_path = r'./checkpoints/'
if os.path.exists(checkpoint_path) is not True:
    os.makedirs(checkpoint_path)



checkpoint_PATH = os.path.join(r'./pretraining_cks', 'pretraining_model_scale_' + str(down) + '.pth.tar')
model_CKPT = torch.load(checkpoint_PATH, map_location='cpu')
finetune_model.load_state_dict(model_CKPT['encoder_state_dict'])
print('loading checkpoint from: ', checkpoint_PATH)


def validate(model, val_loader, val_nrmses):
    pred_HR_SM, HR_SM = [], []
    with torch.no_grad():
        model.eval()
        for step, (LR_img, HR_img, freqs, coils, _, _) in enumerate(val_loader):
            LR_img = LR_img.to(device)
            freqs, coils = freqs.to(device), coils.to(device)
            pred_HR_img = model.predict(LR_img, freqs, coils).cpu().numpy()

            pred_HR_SM.append(pred_HR_img)
            HR_SM.append(HR_img.numpy())

            del LR_img, HR_img, freqs, coils

        pred_HR_SM = np.concatenate(pred_HR_SM, 0) * LR_std + LR_mean
        HR_SM = np.concatenate(HR_SM, 0)

        comp_HR_SM = HR_SM[:, 0, :, :, :] + 1j * HR_SM[:, 1, :, :, :]
        comp_reco_SM = pred_HR_SM[:, 0, :, :, :] + 1j * pred_HR_SM[:, 1, :, :, :]

        vec_reco_SM = comp_reco_SM.reshape(comp_reco_SM.shape[0], 1, -1)
        vec_HR_SM = comp_HR_SM.reshape(comp_HR_SM.shape[0], 1, -1)
        
        # alpha = (vec_reco_SM.conj() * vec_HR_SM).sum((1, 2)) / (vec_reco_SM.conj() * vec_reco_SM).sum((1, 2))
        # vec_reco_SM = vec_reco_SM * alpha.reshape(-1, 1, 1)

        N = vec_reco_SM.shape[-1]
        rmse = np.linalg.norm(vec_reco_SM - vec_HR_SM, 'fro', (1, 2)) / np.sqrt(N)
        val_nrmse = rmse / (np.abs(vec_HR_SM).max((1, 2)) - np.abs(vec_HR_SM).min((1, 2)))
        val_nrmses.append(val_nrmse.mean())

        del pred_HR_img, alpha
        del pred_HR_SM, HR_SM, comp_HR_SM, comp_reco_SM, vec_reco_SM, vec_HR_SM
        # torch.cuda.empty_cache()

        print(epoch, ' iters: ', total_iters, 'val nrmse: ', val_nrmses[-1])


def re_finetune(train_loader):
    model = FreqSResolutionTransformer(image_size=image_size, scale=down, SR_size=SR_size,
                                                patch_size=patch_size,
                                                out_channel=out_channel, kernel_size=kernel_size,
                                                encoder_dim=encoder_dim, encoder_n_layers=encoder_n_layers,
                                                encoder_heads=encoder_heads, encoder_dim_head=encoder_dim_head,
                                                encoder_mlp_dim=encoder_mlp_dim,
                                                dropout=dropout, emb_dropout=emb_dropout, masking_ratio=0.,
                                                loss_type=loss_type, pool=pool, channels=channels, HR_channel=2,
                                                n_conv=n_conv)

    device = torch.device('cuda:0')
    model = model.to(device)
    encoder_layers = list(map(id, finetune_model.encoder.parameters()))
    others_layers = list(filter(lambda p: id(p) not in encoder_layers, finetune_model.parameters()))
    params = [
        {"params": finetune_model.encoder.parameters(), "lr": lr * 0.5},
        {"params": others_layers},
    ]

    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5, betas=(0.9, 0.99))
    checkpoint_PATH = checkpoint_path + \
                      'scale_' + str(down) + \
                      '_epoch_' + str(total_epochs) + \
                      '_kernel_' + str(kernel_size) + \
                      '_patch_size_' + str(patch_size[0]) + \
                      'batch_size_' + str(batch_size) + \
                      '_trans_' + str(encoder_n_layers) + \
                      '_finetune.pth.tar'

    model_CKPT = torch.load(checkpoint_PATH, map_location='cpu')
    model.load_state_dict(model_CKPT['encoder_state_dict'])
    optimizer.load_state_dict(model_CKPT['optimizer'])
    print('loading checkpoint from: ', checkpoint_PATH)

    for param_group in finetune_optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.05

    for _ in range(1):
        for step, (LR_img, HR_img, freqs, coils) in enumerate(train_loader):
            LR_img, HR_img = LR_img.to(device), HR_img.to(device)
            freqs, coils = freqs.to(device), coils.to(device)
            loss = model(LR_img, HR_img, freqs, coils)

            loss = loss / LR_img.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_name = checkpoint_path + \
                'scale_' + str(down) + \
                '_epoch_' + str(total_epochs) + \
                '_kernel_' + str(kernel_size) + \
                '_patch_size_' + str(patch_size[0]) + \
                'batch_size_' + str(batch_size) + \
                '_trans_' + str(encoder_n_layers) + \
                '_finetune.pth.tar'
    torch.save({'epoch': epoch, 'encoder_state_dict': finetune_model.state_dict(),
                'optimizer': finetune_optimizer.state_dict()}, save_name
               )


# m = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
total_iters = start_epoch * (4129 // batch_size + 1)
ts = time.time()
for epoch in range(start_epoch + 1, total_epochs + 1):
    train_loss = 0.0
    val_loss = 0.0

    finetune_model.train()
    for step, (LR_img, HR_img, freqs, coils) in enumerate(train_loader):
        total_iters += 1
        adjust_learning_rate(finetune_optimizer, lr, step / len(train_loader) + epoch - 1, 10, 100)
        LR_img, HR_img = LR_img.to(device), HR_img.to(device)
        freqs, coils = freqs.to(device), coils.to(device)
        loss = finetune_model(LR_img, HR_img, freqs, coils)

        loss = loss / LR_img.shape[0]

        finetune_optimizer.zero_grad()
        loss.backward()
        finetune_optimizer.step()

        train_loss += loss.item()

    train_losses.append(train_loss / (step + 1))
    train_te = time.time()

    print(epoch, 'train loss: ', train_losses[-1], 'cost time: ', (train_te - ts) / 60)

    if epoch == total_epochs:
        save_name = checkpoint_path + \
                     'scale_' + str(down) + \
                     '_epoch_' + str(epoch) + \
                     '_kernel_' + str(kernel_size) + \
                     '_patch_size_' + str(patch_size[0]) + \
                     'batch_size_' + str(batch_size) + \
                     '_trans_' + str(encoder_n_layers) + \
                     '_finetune.pth.tar'
        torch.save({'epoch': epoch, 'encoder_state_dict': finetune_model.state_dict(),
                    'optimizer': finetune_optimizer.state_dict()}, save_name
                    )

re_finetune(train_loader)

