# Modified from https://github.com/JingyunLiang/SwinIR
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.

import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import numpy as np
from basicsr.archs.SwinIR_block import BasicLayer,PatchEmbed,PatchUnEmbed


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class BAFM(nn.Module):
    def __init__(self, dim,num_blocks,SA_Api):
        super().__init__()
        self.dim = dim
        self.SA_Api =SA_Api
        self.cat_num = num_blocks + 2
        self.qk = nn.Sequential(
            nn.Conv3d(self.cat_num, self.cat_num, 1, 1, 0),
            nn.Conv3d(self.cat_num, self.cat_num, 3, 1, 1, groups=self.cat_num),
        )
        self.v = nn.Sequential(
            nn.Conv3d(self.cat_num, self.cat_num, 1, 1, 0),
            nn.Conv3d(self.cat_num, self.cat_num, 3, 1, 1, groups=self.cat_num),
        )
        self.projcet_out = nn.Conv3d(self.cat_num, self.cat_num, 1, 1, 0)
        #self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Conv3d(self.cat_num, 1, kernel_size=3, stride=1, padding=1)
        if self.SA_Api==False:
            self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

    # b n c h w -> b c h w
    def forward(self, x):
        b, n, c, h, w = x.shape
        q = self.qk(x)  # b n c h w
        q = rearrange(q, "b n c h w -> b n (c h w)")
        v = self.v(x)
        v = rearrange(v, "b n c h w -> b n (c h w)")

        # 使用torch api
        if self.SA_Api:
            out = torch.nn.functional.scaled_dot_product_attention(q,q,v)
        else:
            att = self.alpha * (q @ q.transpose(-2, -1))  # b n n
            att = att.softmax(dim=-1)
            out = att @ v
        out = rearrange(out, "b n (c h w) ->b n c h w", b=b, n=n, c=c, h=h, w=w)
        out = self.projcet_out(out)
        out = self.gamma * out + x
        #out = out.mean(1)
        out = self.conv(out)
        out = out.squeeze(1)
        return out

class FIFB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_equi = nn.Sequential(
            nn.Conv2d(2*dim, dim,1,1,0),
            nn.GELU(),
        )
        self.conv_cube = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1,1,0),
            nn.GELU(),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1,1,0),
            nn.GELU()
        )


    def forward(self, f_erp, f_cmp):
        f_cat = torch.concat([f_erp, f_cmp], dim=1)
        f_cat = self.conv_cat(f_cat)
        f_cmp = torch.concat([f_cat, f_cmp], dim=1)
        f_cmp = self.conv_cat(f_cmp)
        f_erp = torch.concat([f_cat, f_erp], dim=1)
        f_erp = self.conv_cat(f_erp)
        return f_erp,f_cmp,f_cat


class Equirec2Cube(nn.Module):
    def __init__(self, cube_dim, equ_h, FoV=90.0):
        super().__init__()
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = (
            np.array(
                [
                    [0, -180.0, 0],
                    [90.0, 0, 0],
                    [0, 0, 0],
                    [0, 90, 0],
                    [0, -90, 0],
                    [-90, 0, 0],
                ],
                np.float32,
            )
            / 180.0
            * np.pi
        )
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        grids = self._getCubeGrid()

        for i, grid in enumerate(grids):
            self.register_buffer("grid_%d" % i, grid)

    def _getCubeGrid(self):
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        # self.grids = []
        grids = []
        for _, R in enumerate(self.R_lst):
            tmp = (
                xyz @ R
            )  # Don't need to transpose since we are doing it for points not for camera
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            grids.append(torch.FloatTensor(lonlat[None, ...]))

        return grids

    def forward(self, batch, mode="bilinear"):
        [_, _, h, w] = batch.shape

        assert h == self.equ_h and w == self.equ_w
        assert mode in ["nearest", "bilinear"]

        out = []
        for i in range(6):
            grid = getattr(self, "grid_%d" % i)
            grid = grid.repeat(batch.shape[0], 1, 1, 1)
            sample = F.grid_sample(batch, grid, mode=mode, align_corners=True)
            out.append(sample)
        out = torch.cat(out, dim=0)
        final_out = []
        for i in range(batch.shape[0]):
            final_out.append(out[i :: batch.shape[0], ...])
        final_out = torch.cat(final_out, dim=0)
        return final_out


class Cube2Equirec(nn.Module):
    def __init__(self, cube_length, equ_h):
        super().__init__()
        self.cube_length = cube_length
        self.equ_h = equ_h
        equ_w = equ_h * 2
        self.equ_w = equ_w
        theta = (np.arange(equ_w) / (equ_w - 1) - 0.5) * 2 * np.pi
        phi = (np.arange(equ_h) / (equ_h - 1) - 0.5) * np.pi

        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

        planes = np.asarray(
            [
                [0, 0, 1, 1],  # z = -1
                [0, 1, 0, -1],  # y =  1
                [0, 0, 1, -1],  # z =  1
                [1, 0, 0, 1],  # x = -1
                [1, 0, 0, -1],  # x =  1
                [0, 1, 0, 1],  # y = -1
            ]
        )
        r_lst = (
            np.array(
                [
                    [0, 1, 0],
                    [0.5, 0, 0],
                    [0, 0, 0],
                    [0, 0.5, 0],
                    [0, -0.5, 0],
                    [-0.5, 0, 0],
                ]
            )
            * np.pi
        )
        f = cube_length / 2.0
        self.K = np.array(
            [
                [f, 0, (cube_length - 1) / 2.0],
                [0, f, (cube_length - 1) / 2.0],
                [0, 0, 1],
            ]
        )
        self.R_lst = [cv2.Rodrigues(x)[0] for x in r_lst]

        masks, XYs = self._intersection(xyz, planes)

        for i in range(6):
            self.register_buffer("mask_%d" % i, masks[i])
            self.register_buffer("XY_%d" % i, XYs[i])

    def forward(self, x, mode="bilinear"):
        assert mode in ["nearest", "bilinear"]
        assert x.shape[0] % 6 == 0
        equ_count = x.shape[0] // 6
        equi = torch.zeros(
            equ_count, x.shape[1], self.equ_h, self.equ_w, device=x.device
        )
        for i in range(6):
            now = x[i::6, ...]
            mask = getattr(self, "mask_%d" % i)
            mask = mask[None, ...].repeat(equ_count, x.shape[1], 1, 1)

            XY = (
                getattr(self, "XY_%d" % i)[None, None, :, :].repeat(equ_count, 1, 1, 1)
                / (self.cube_length - 1)
                - 0.5
            ) * 2
            sample = F.grid_sample(now, XY, mode=mode, align_corners=True)[..., 0, :]
            equi[mask] = sample.view(-1)

        return equi

    def _intersection(self, xyz, planes):
        abc = planes[:, :-1]

        depth = -planes[:, 3][None, None, ...] / np.dot(xyz, abc.T)
        depth[depth < 0] = np.inf
        arg = np.argmin(depth, axis=-1)
        depth = np.min(depth, axis=-1)

        pts = depth[..., None] * xyz

        mask_lst = []
        mapping_XY = []
        for i in range(6):
            mask = arg == i
            mask = np.tile(mask[..., None], [1, 1, 3])

            XY = np.dot(np.dot(pts[mask].reshape([-1, 3]), self.R_lst[i].T), self.K.T)
            XY = np.clip(XY[..., :2].copy() / XY[..., 2:], 0, self.cube_length - 1)
            mask_lst.append(mask[..., 0])
            mapping_XY.append(XY)
        mask_lst = [torch.BoolTensor(x) for x in mask_lst]
        mapping_XY = [torch.FloatTensor(x) for x in mapping_XY]

        return mask_lst, mapping_XY

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class BPOSR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=(64,64),
                 roll_r = 3,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 cmp_window_size=(8,8),
                 erp_window_size =(8,16),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(BPOSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        erp_img_size = img_size
        cmp_img_size = (img_size[0]//2,img_size[0]//2)
        self.roll_r =roll_r
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler

        self.erp_conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.cmp_conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.E2C = Equirec2Cube(cube_dim=img_size[0]// 2, equ_h=img_size[0], FoV=90.0)
        self.C2E = Cube2Equirec(cube_length=img_size[0] // 2, equ_h=img_size[0])

        self.erp_patch_embed = PatchEmbed(
            img_size=erp_img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        erp_num_patches = self.erp_patch_embed.num_patches
        erp_patches_resolution = self.erp_patch_embed.patches_resolution
        self.erp_patches_resolution = erp_patches_resolution

        self.erp_patch_unembed = PatchUnEmbed(
            img_size=erp_img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.erp_absolute_pos_embed = nn.Parameter(torch.zeros(1, erp_num_patches, embed_dim))
            trunc_normal_(self.erp_absolute_pos_embed, std=.02)


        self.cmp_patch_embed = PatchEmbed(
            img_size=cmp_img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        cmp_num_patches = self.cmp_patch_embed.num_patches
        cmp_patches_resolution = self.cmp_patch_embed.patches_resolution
        self.cmp_patches_resolution = cmp_patches_resolution

        self.cmp_patch_unembed = PatchUnEmbed(
            img_size=cmp_img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.cmp_absolute_pos_embed = nn.Parameter(torch.zeros(1, cmp_num_patches, embed_dim))
            trunc_normal_(self.cmp_absolute_pos_embed, std=.02)


        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.erp_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            erp_layer = RSTB(
                dim=embed_dim,
                input_resolution=(erp_patches_resolution[0], erp_patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=erp_window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=erp_img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.erp_layers.append(erp_layer)

        self.cmp_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            cmp_layer = RSTB(
                dim=embed_dim,
                input_resolution=(cmp_patches_resolution[0], cmp_patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=cmp_window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=erp_img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.cmp_layers.append(cmp_layer)

        self.fifb_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            fifb_layer =FIFB(embed_dim)
            self.fifb_layers.append(fifb_layer)

        self.bafm = BAFM(embed_dim,self.num_layers,False)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (erp_patches_resolution[0], erp_patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward_features(self, erp,cmp):

        for i,(erp_layer,cmp_layer,fifb_layer) in enumerate(zip(self.erp_layers, self.cmp_layers,self.fifb_layers)):
            erp_size = (erp.shape[2], erp.shape[3])
            erp = self.erp_patch_embed(erp)
            erp = erp_layer(erp, erp_size)
            erp = self.erp_patch_unembed(erp, erp_size)

            cmp = torch.roll(cmp, shifts=int(cmp.shape[-1]*(i%self.roll_r)//self.roll_r ), dims=-1)
            cmp = self.E2C(cmp)
            cmp_size = (cmp.shape[2], cmp.shape[3])
            cmp = self.cmp_patch_embed(cmp)
            cmp = cmp_layer(cmp, cmp_size)
            cmp = self.cmp_patch_unembed(cmp, cmp_size)
            cmp = self.C2E(cmp)
            cmp = torch.roll(cmp, shifts=int(-cmp.shape[-1]*(i%self.roll_r)//self.roll_r ), dims=-1)
            erp ,cmp ,cat = fifb_layer(erp,cmp)
            if i == 0:
                fin_cat = cat.unsqueeze(1)
            else:
                fin_cat = torch.concat([cat.unsqueeze(1), fin_cat], dim=1)
        fin_cat = torch.concat([fin_cat, cmp.unsqueeze(1), erp.unsqueeze(1)], dim=1)
        out = self.bafm(fin_cat)
        return out

    def forward(self, x):

        if self.upsampler == 'pixelshuffle':
            # for classical SR

            erp = self.erp_conv_first(x)
            cmp = self.E2C(x)
            cmp = self.cmp_conv_first(cmp)
            cmp =self.C2E(cmp)
            x = self.conv_after_body(self.forward_features(erp,cmp))+ erp + cmp
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            erp = self.erp_conv_first(erp)
            cmp = self.E2C(erp)
            cmp = self.cmp_conv_first(cmp)
            cmp =self.C2E(cmp)
            x = self.conv_after_body(self.forward_features(erp,cmp))+ erp + cmp
            x = self.upsample(x)

        return x



if __name__ == '__main__':
    device = torch.device('cuda:3')
    torch.cuda.empty_cache()

    h = 1024//8
    w = 2048//8
      
    model = BPOSR(
  upscale= 8,
  in_chans= 3,
  img_size=[128,256],
  cmp_window_size=[8,8],
  erp_window_size=[4,16],
  roll_r=3,
  img_range=1.,
  depths=[6,6,6,6],
  embed_dim=60,
  num_heads=[6,6,6,6],
  mlp_ratio= 2,
  upsampler='pixelshuffle',
  resi_connection='1conv',
).to(device)

    #print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, h, w)).to(device)
    model.eval()

    with torch.no_grad():
        for i in range(100):
            y = model(x)

    print(y.shape)
    print(torch.cuda.memory_summary(device=device))
    t = torch.cuda.max_memory_allocated(device=device)
    print(t/1024/1024/1024)

    t = torch.cuda.memory_allocated(device=device)
    print(t/1024/1024/1024)
    #summary(model, (3, h, w))