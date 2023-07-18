import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import init
from torch.nn import functional as F
import numpy as np


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x shape (batch, n_patch, in_dim)
        # out shape is the same
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def get_attn(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        return attn
    


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PatchEmbedder(nn.Module):
    def __init__(self, patch_dim, num_patches, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        # img  is rearranged
        # shape is (batch, H * W * D / p^3, patch_dim)
        # masked_indices (batch, H*W*D*mask_ration)
        device = img.device

        # shape is (batch, D* H * W / p^3, dim)
        patches_emb = self.to_patch_embedding(img)

        batch, num_patches, _ = patches_emb.shape
        # we will not ad cls token also
        x = patches_emb + self.pos_embedding[:, 1:(num_patches + 1)]

        return x


class Encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim,
                 n_layers, heads, mlp_dim, pool='cls', channels=2,
                 dim_head=64, dropout=0., emb_dropout=0.):
        '''
        image_size and patch_size are 3D tuples, the default of patch_size is (1, 1, 1)
        dim is the initial patch dimensions
        '''

        super().__init__()

        image_depth, image_height, image_width = image_size
        patch_depth, patch_height, patch_width = patch_size

        # Image dimensions must be divisible by the patch size.
        assert image_height % patch_height == 0 and \
               image_width % patch_width == 0 and \
               image_depth % patch_depth == 0

        num_patches = (image_height // patch_height) * \
                      (image_width // patch_width) * \
                      (image_depth // patch_depth)

        patch_dim = channels * patch_height * patch_width * patch_depth
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.ReArrange = nn.Sequential(
            Rearrange('b c (d p3) (h p1) (w p2) -> b (d h w) (p3 p1 p2 c)',
                      p1=patch_height, p2=patch_width, p3=patch_depth)
        )

        self.patch_embedder = PatchEmbedder(patch_dim, num_patches, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, n_layers, heads, dim_head, mlp_dim, dropout)

    def forward(self, img, additional_emb=None):
        # img shape is (batch, channel, D, H, W)

        device = img.device
        batch = img.shape[0]

        # patches partition, shape (b, 2, D, H, W) -> (b, DHW / p^3, 2*p^3)
        # p is the patch size
        x = self.ReArrange(img)

        # get patch embedding
        # shape (b, DHW / p^3, 2*p^3) -> (b, DHW / p^3, dim)
        x = self.patch_embedder(x)

        # additional emb, each emb shape is (b, 1, dim)
        # assume len(additional_emb) = l
        # output shape (b, DHW / p^3 + l, dim)

        if additional_emb is not None:
            for emb in additional_emb:
                x = torch.cat([x, emb], 1)
                # x = x + p * emb

        x = self.dropout(x)

        # transformer layer doesn't change shape
        # shape (b, DHW / p^3 + l, dim)
        x = self.transformer(x)

        return x


class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        # (b, c, D, H, W) -> (b, c/r^3, r*D, r*H, r*W)
        # r is scale factor, c % r^3 is required

        batch_size, channels, in_depth, in_height, in_width = input.size()
        assert channels % (self.scale ** 3) == 0
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale,
                                             in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class UpSampler(nn.Module):
    def __init__(self, scale, in_channel, out_channel, kernel_size, n_conv):
        super().__init__()
        self.upsampling = nn.Sequential(
            PixelShuffle3d(scale=scale)
        )

        self.convs = nn.ModuleList([
                      nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size,
                                stride=1, padding=(kernel_size - 1) // 2),
                      nn.PReLU(),
                      nn.BatchNorm3d(out_channel),

        ])
        # for _ in range(n_conv):

        for _ in range(n_conv-1):

            self.convs.append(
                nn.Sequential(nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size,
                                        stride=1, padding=(kernel_size - 1) // 2),
                              nn.PReLU(),
                              nn.BatchNorm3d(out_channel),


            ))


    def forward(self, img, additional_emb=None):
        # img shape (b, dim, D, H, W)
        # dim is considered as channels

        # up-sampling (b, dim, D, H, W) -> (b, dim, r*D, r*H, r*W)
        x = self.upsampling(img)

        if additional_emb is not None:
            for emb in additional_emb:
                x = torch.cat([x, emb], 1)

        for step, conv in enumerate(self.convs):
            if step == 0:
                x = conv(x)
            else:
                x = conv(x) + x

        return x



class DataConsistencyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.p = nn.Parameter(torch.rand(1), requires_grad=True)
        self.p = 0.5

    def forward(self, pred_img, samp_img, mask, p=None):
        if p is None:
            p = self.p

        return mask * (samp_img * p + pred_img) / (1 + p) + pred_img * (~mask)



class FreqSResolutionTransformer(nn.Module):
    def __init__(self, *, image_size, scale, SR_size, patch_size, out_channel, kernel_size,
                 encoder_dim, encoder_n_layers, encoder_heads, encoder_dim_head, encoder_mlp_dim,
                 dropout=0., emb_dropout=0., masking_ratio=0., loss_type='l2', pool='cls', channels=3,
                 HR_channel=2, n_conv=3):
        super().__init__()
        self.scale = scale
        self.SR_size = SR_size
        self.isMasking = masking_ratio > 0
        self.encoder = Encoder(image_size=image_size, patch_size=patch_size, dim=encoder_dim, n_layers=encoder_n_layers,
                               heads=encoder_heads, mlp_dim=encoder_mlp_dim, pool=pool, channels=channels,
                               dim_head=encoder_dim_head, dropout=dropout, emb_dropout=emb_dropout)
        self.token_size = (image_size[0] // patch_size[0],
                           image_size[1] // patch_size[1],
                           image_size[2] // patch_size[2])



        self.token2image = nn.Sequential(
            Rearrange('b (d h w) c -> b c d h w',
                      d=self.token_size[0], h=self.token_size[1], w=self.token_size[2]),
            PixelShuffle3d(scale=2)
            # nn.ConvTranspose3d(encoder_dim, out_channel, kernel_size=2, stride=2)
            if patch_size[0] in [2, 4] else nn.Identity()
        )

        self.image2token = Rearrange('b c p1 p2 p3 -> b (p1 p2 p3) c')

        self.freqEmbeder = nn.Linear(1, encoder_dim)
        self.coilEmbeder = nn.Embedding(3, encoder_dim)
        self.freqPosEmbedding = nn.Parameter(torch.randn(1, 2, encoder_dim), requires_grad=True)
        # init.kaiming_uniform_(self.freqPosEmbedding)


        if scale in [2, 3]:
            in_channel = encoder_dim // (scale * patch_size[0]) ** 3
            self.SRdecoder = UpSampler(scale, in_channel, out_channel, kernel_size, n_conv)
        else:
            assert scale == 4
            half_in_channel = encoder_dim // (2 * patch_size[0]) ** 3
            in_channel = out_channel // (2 * patch_size[0]) ** 3
            self.half_SRdecoder = UpSampler(2, half_in_channel, out_channel, kernel_size, n_conv)
            self.SRdecoder = UpSampler(2, in_channel, out_channel, kernel_size, n_conv)
            self.half_residual_conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv3d(HR_channel, out_channel, kernel_size=kernel_size,
                          stride=1, padding=(kernel_size - 1) // 2),
                nn.PReLU(),
                nn.BatchNorm3d(out_channel),
            )

        assert loss_type in ['l2', 'l1']
        self.loss_func = nn.MSELoss(reduction='sum')
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')

        self.mask = torch.zeros(1, 1, SR_size[0], SR_size[0], SR_size[0])
        self.mask[:, :, 1::scale, 1::scale, 1::scale] = 1

        self.mask = self.mask.bool()

        self.dcLayer = DataConsistencyLayer()
        self.pred = nn.Conv3d(out_channel, HR_channel, kernel_size=1, stride=1)

        self.residual_conv = nn.Sequential(
                                      nn.Upsample(scale_factor=scale),
                                      nn.Conv3d(HR_channel, out_channel, kernel_size=kernel_size,
                                                stride=1, padding=(kernel_size-1) // 2),
                                      nn.PReLU(),
                                      nn.BatchNorm3d(out_channel)
                                      )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, LR_img, HR_img, freqs, coils):
        # LR_img shape is (batch, channel, D, H, W)
        # HR_img shape is (batch, channel, SR_size)
        device = LR_img.device
        batch = LR_img.shape[0]

        self.mask = self.mask.to(device)
        samp_img = torch.zeros(batch, 2, self.SR_size[0], self.SR_size[1], self.SR_size[2]).to(device)
        samp_img[:, :, self.mask.squeeze()] = LR_img.view(batch, 2, -1)


        # (b, 1) -> (b, 1, encoder_dim)
        freq_emb = self.freqEmbeder(freqs).view(batch, 1, -1) + self.freqPosEmbedding[:, 0, :]
        coil_emb = self.coilEmbeder(coils).view(batch, 1, -1) + self.freqPosEmbedding[:, 1, :]


        encoded_tokens = self.encoder(LR_img, [freq_emb, coil_emb])
        encoded_tokens = encoded_tokens[:, :-2, :]

        # (b, DHW / p ^ 3, encoder_dim) -> (b, encoder_dim / p ^ 3, D, H, W)
        encoded_tokens = self.token2image(encoded_tokens)

        # predict HR SM
        # (b, tar_channel, SR_size)
        if self.scale in [2, 3]:
            # encoded_tokens = torch.cat([encoded_tokens, shallow_feat], 1)
            decoded_tokens = self.SRdecoder(encoded_tokens)
        else:
            half_decoded_tokens = self.half_SRdecoder(encoded_tokens) + self.half_residual_conv(LR_img)
            decoded_tokens = self.SRdecoder(half_decoded_tokens)


        pred_HR_SM = self.pred(decoded_tokens + self.residual_conv(LR_img))

        # caculate loss
        loss = self.loss_func(pred_HR_SM, HR_img) / np.prod(pred_HR_SM.shape[1:])

        return loss

    def predict(self, LR_img, freqs, coils, p=None):

        # LR_img shape is (batch, channel, D, H, W)
        batch = LR_img.shape[0]
        device = LR_img.device

        self.mask = self.mask.to(device)
        samp_img = torch.zeros(batch, 2, self.SR_size[0], self.SR_size[1], self.SR_size[2]).to(device)
        samp_img[:, :, self.mask.squeeze()] = LR_img.view(batch, 2, -1)

        # (b, 1) -> (b, 1, encoder_dim)
        freq_emb = self.freqEmbeder(freqs).view(batch, 1, -1) + self.freqPosEmbedding[:, 0, :]
        coil_emb = self.coilEmbeder(coils).view(batch, 1, -1) + self.freqPosEmbedding[:, 1, :]

        encoded_tokens = self.encoder(LR_img, [freq_emb, coil_emb])
        encoded_tokens = encoded_tokens[:, :-2, :]

        # (b, DHW / p ^ 3, encoder_dim) -> (b, encoder_dim / p ^ 3, D, H, W)
        encoded_tokens = self.token2image(encoded_tokens)

        # predict HR SM
        # (b, tar_channel, SR_size)
        if self.scale in [2, 3]:
            # encoded_tokens = torch.cat([encoded_tokens, shallow_feat], 1)
            decoded_tokens = self.SRdecoder(encoded_tokens)
        else:
            half_decoded_tokens = self.half_SRdecoder(encoded_tokens) + self.half_residual_conv(LR_img)
            decoded_tokens = self.SRdecoder(half_decoded_tokens)

        pred_HR_SM = self.pred(decoded_tokens + self.residual_conv(LR_img))
        pred_HR_SM = self.dcLayer(pred_HR_SM, samp_img, self.mask, p=p)

        return pred_HR_SM
