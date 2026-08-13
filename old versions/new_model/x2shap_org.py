"""
X2Shape: CT-free 3D multi-organ reconstruction with biplanar X-rays
完整还原论文架构 - 4层U-Net结构 (优化版)
直接使用 nn.InstanceNorm，无冗余包装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✓ 使用官方 mamba_ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    raise ImportError("请先安装 mamba_ssm")


# ============================================================
# 1. 2D MambaVision 编码器 (完整版)
# ============================================================

class LayerNorm2d(nn.Module):
    """2D Layer Normalization (用于MambaVision)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MambaVisionMixer(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        self.inner_dim = dim * expand
        self.in_proj = nn.Linear(dim, 2 * self.inner_dim)
        self.conv1d = nn.Conv1d(self.inner_dim, self.inner_dim, 3, padding=1, groups=self.inner_dim)
        self.mamba = Mamba(d_model=self.inner_dim, d_state=16, d_conv=4, expand=1)
        self.out_proj = nn.Linear(self.inner_dim, dim)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = rearrange(x, 'b c h w -> b (h w) c')
            return_4d = True
        else:
            B, N, C = x.shape
            x_flat = x
            return_4d = False

        xz = self.in_proj(x_flat)
        x_ssm, x_conv = xz.chunk(2, dim=-1)

        x_ssm = rearrange(x_ssm, 'b l d -> b d l')
        x_ssm = self.conv1d(x_ssm)
        x_ssm = rearrange(x_ssm, 'b d l -> b l d')
        x_ssm = self.mamba(x_ssm)

        x_conv = F.silu(x_conv)
        x_out = self.out_proj(x_ssm + x_conv)

        if return_4d:
            return rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            return x_out


class SelfAttention2D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = rearrange(x, 'b c h w -> b (h w) c')
            return_4d = True
        else:
            B, N, C = x.shape
            x_flat = x
            return_4d = False
            H = W = int(math.sqrt(N))

        qkv = self.qkv(x_flat).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if return_4d:
            return rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            return x_out


class MambaVisionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, use_attention=False, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if use_attention:
            self.mixer = SelfAttention2D(dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        else:
            self.mixer = MambaVisionMixer(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')

        x_flat = x_flat + self.mixer(self.norm1(x_flat))
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)


class MambaVisionEncoder2D(nn.Module):
    def __init__(self, in_chans=1, dims=[32, 64, 128, 256], depths=[1, 1, 2, 1],
                 use_attention_layers=[False, False, True, True], dropout=0.0):
        super().__init__()
        self.dims = dims
        self.num_scales = len(dims)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        current_dim = dims[0]

        for i in range(self.num_scales):
            stage_blocks = nn.Sequential()
            for j in range(depths[i]):
                use_attn = use_attention_layers[i] if i < len(use_attention_layers) else False
                stage_blocks.append(
                    MambaVisionBlock(current_dim, use_attention=use_attn,
                                   num_heads=8, dropout=dropout)
                )
            self.stages.append(stage_blocks)

            if i < self.num_scales - 1:
                self.stages.append(
                    nn.Sequential(
                        nn.Conv2d(current_dim, dims[i+1], kernel_size=2, stride=2),
                        LayerNorm2d(dims[i+1]),
                        nn.GELU()
                    )
                )
                current_dim = dims[i+1]

    def forward(self, x):
        x = self.stem(x)
        features = []

        for layer in self.stages:
            x = layer(x)
            if isinstance(layer, nn.Sequential) and len(layer) > 0:
                if not isinstance(layer[0], nn.Conv2d):
                    features.append(x)

        return features


# ============================================================
# 2. 体积反投影模块 (VBP) - 直接使用 nn.InstanceNorm
# ============================================================

class LocalFeatureExpand(nn.Module):
    def __init__(self, in_channels, out_channels, volume_shape):
        super().__init__()
        self.volume_shape = volume_shape

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.refine_3d = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, f_2d, proj_type):
        B, C, H, W = f_2d.shape
        X_dim, Y_dim, Z_dim = self.volume_shape

        f_proc = self.expand_conv(f_2d)

        if proj_type == 'ap':
            f_3d = f_proc[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)
        else:
            f_3d = f_proc[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)

        f_3d = self.refine_3d(f_3d)
        return f_3d


class GlobalFeatureExpand(nn.Module):
    def __init__(self, feat_dim, embed_dim=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, feat_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, f_ap_3d, f_lat_3d):
        B, C, X_dim, Y_dim, Z_dim = f_ap_3d.shape
        N = X_dim * Y_dim * Z_dim

        f_ap_flat = rearrange(f_ap_3d, 'b c x y z -> b (x y z) c')
        f_lat_flat = rearrange(f_lat_3d, 'b c x y z -> b (x y z) c')

        x_coords = torch.linspace(-1, 1, X_dim, device=f_ap_3d.device)
        y_coords = torch.linspace(-1, 1, Y_dim, device=f_ap_3d.device)
        z_coords = torch.linspace(-1, 1, Z_dim, device=f_ap_3d.device)

        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        pos_encoding = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1)

        pos_embed = self.pos_mlp(pos_encoding)

        concat_feat = torch.cat([f_ap_flat, f_lat_flat, pos_embed], dim=-1)
        feat_embed = self.fusion_mlp(concat_feat)

        attn_scores = torch.norm(feat_embed, dim=-1, keepdim=True)
        attn_weights = F.softmax(attn_scores, dim=1)

        weighted_sum = torch.sum(attn_weights * feat_embed, dim=1, keepdim=True)

        global_feat = self.output_mlp(weighted_sum)
        global_feat = global_feat.expand(-1, N, -1)

        global_feat_3d = rearrange(global_feat, 'b (x y z) c -> b c x y z',
                                   x=X_dim, y=Y_dim, z=Z_dim)

        return global_feat_3d


class VolumetricBackprojection(nn.Module):
    def __init__(self, multi_scale_channels, out_channels, volume_shape=(64, 64, 64),
                 embed_dim=64, dropout=0.0):
        super().__init__()
        self.multi_scale_channels = multi_scale_channels
        self.out_channels = out_channels
        self.volume_shape = volume_shape
        self.num_scales = len(multi_scale_channels)
        self.embed_dim = embed_dim

        self.fusion_convs = nn.ModuleList()
        for ch in multi_scale_channels:
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),  # 直接使用
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),  # 直接使用
                    nn.LeakyReLU(0.1, inplace=True)
                )
            )

        self.local_expand = LocalFeatureExpand(out_channels, out_channels, volume_shape)
        self.global_expand = GlobalFeatureExpand(out_channels, embed_dim)

        self.final_fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True)
        )

        X, Y, Z = volume_shape
        self.position_weights = nn.Parameter(torch.ones(1, 1, X, Y, Z))

    def forward(self, f_multi_scale, f_other_view_3d=None):
        B = f_multi_scale[0].shape[0]
        X_dim, Y_dim, Z_dim = self.volume_shape

        fused_2d = []
        for i, f in enumerate(f_multi_scale):
            if f.shape[-2:] != (X_dim, Z_dim):
                f_up = F.interpolate(f, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)
            else:
                f_up = f
            f_proj = self.fusion_convs[i](f_up)
            fused_2d.append(f_proj)

        fused_2d = sum(fused_2d)

        local_feat = self.local_expand(fused_2d, 'ap')

        if f_other_view_3d is not None:
            global_feat = self.global_expand(local_feat, f_other_view_3d)
        else:
            global_feat = local_feat

        combined = torch.cat([local_feat, global_feat], dim=1)
        out = self.final_fusion(combined)

        pos_weight = self.position_weights.expand(B, -1, -1, -1, -1)
        out = out * torch.sigmoid(pos_weight)

        return out


# ============================================================
# 3. 3D Encoder (4层) - 直接使用 nn.InstanceNorm
# ============================================================

class Encoder3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        feat = x
        if self.downsample:
            x = self.pool(x)
        return x, feat


class Encoder3D(nn.Module):
    def __init__(self, in_chans, channels=[32, 64, 128, 256]):
        super().__init__()
        self.channels = channels

        self.blocks = nn.ModuleList()
        curr_ch = in_chans

        for i, out_ch in enumerate(channels):
            downsample = (i < len(channels) - 1)
            self.blocks.append(Encoder3DBlock(curr_ch, out_ch, downsample))
            curr_ch = out_ch

    def forward(self, x):
        features = []
        for block in self.blocks:
            x, feat = block(x)
            features.append(feat)
        return features[::-1]


# ============================================================
# 4. Cross-Mamba 模块 (4层)
# ============================================================

class CrossMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm_ap = nn.LayerNorm(dim)
        self.norm_lat = nn.LayerNorm(dim)
        self.mamba_ap = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_lat = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.gate_ap = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.gate_lat = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())

    def forward(self, f_ap, f_lat):
        B, C, X_dim, Y_dim, Z_dim = f_ap.shape

        f_ap_flat = rearrange(f_ap, 'b c x y z -> b (x y z) c')
        f_lat_flat = rearrange(f_lat, 'b c x y z -> b (x y z) c')

        y1 = self.mamba_ap(self.norm_ap(f_ap_flat))
        g1 = self.gate_ap(self.norm_ap(f_lat_flat))
        out1 = rearrange(y1 * g1, 'b (x y z) c -> b c x y z', x=X_dim, y=Y_dim, z=Z_dim)

        y2 = self.mamba_lat(self.norm_lat(f_lat_flat))
        g2 = self.gate_lat(self.norm_ap(f_ap_flat))
        out2 = rearrange(y2 * g2, 'b (x y z) c -> b c x y z', x=X_dim, y=Y_dim, z=Z_dim)

        return out1 + out2


# ============================================================
# 5. U-Net 解码器 (带特征拼接)
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # 直接使用
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x, skip_feat):
        x = self.upsample(x)

        if x.shape[-3:] != skip_feat.shape[-3:]:
            x = F.interpolate(x, size=skip_feat.shape[-3:],
                            mode='trilinear', align_corners=False)

        x = torch.cat([x, skip_feat], dim=1)
        x = self.conv(x)

        return x


class UNetDecoder(nn.Module):
    def __init__(self, channels=[256, 128, 64, 32]):
        super().__init__()

        self.decoder_0 = DecoderBlock(
            in_channels=channels[0], skip_channels=channels[1], out_channels=channels[1]
        )
        self.decoder_1 = DecoderBlock(
            in_channels=channels[1], skip_channels=channels[2], out_channels=channels[2]
        )
        self.decoder_2 = DecoderBlock(
            in_channels=channels[2], skip_channels=channels[3], out_channels=channels[3]
        )

    def forward(self, x, skip_features):
        x = self.decoder_0(x, skip_features[1])
        x = self.decoder_1(x, skip_features[2])
        x = self.decoder_2(x, skip_features[3])
        return x


# ============================================================
# 6. 完整的 X2Shape 模型
# ============================================================

class X2Shape(nn.Module):
    def __init__(self, img_size=256, in_chans=1, num_classes=1,
                 dims_2d=[32, 64, 128, 256], depths_2d=[1, 1, 2, 1],
                 dims_3d=[32, 64, 128, 256], vbp_output_channels=64,
                 vbp_embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.num_levels = len(dims_3d)

        vol_sizes = [img_size // 4, img_size // 4, img_size // 4]

        # ========== 1. 两个独立的2D编码器 ==========
        use_attn_layers = [False, False, True, True]
        self.encoder_2d_ap = MambaVisionEncoder2D(
            in_chans, dims_2d, depths_2d,
            use_attention_layers=use_attn_layers
        )
        self.encoder_2d_lat = MambaVisionEncoder2D(
            in_chans, dims_2d, depths_2d,
            use_attention_layers=use_attn_layers
        )

        # ========== 2. VBP模块 ==========
        self.vbp_ap = VolumetricBackprojection(
            dims_2d, vbp_output_channels, vol_sizes,
            embed_dim=vbp_embed_dim
        )
        self.vbp_lat = VolumetricBackprojection(
            dims_2d, vbp_output_channels, vol_sizes,
            embed_dim=vbp_embed_dim
        )

        # ========== 3. 两个独立的3D编码器 ==========
        self.encoder_3d_ap = Encoder3D(vbp_output_channels, dims_3d)
        self.encoder_3d_lat = Encoder3D(vbp_output_channels, dims_3d)

        # ========== 4. Cross-Mamba模块 (4个) ==========
        self.cross_mambas = nn.ModuleList([
            CrossMamba(dims_3d[3]),
            CrossMamba(dims_3d[2]),
            CrossMamba(dims_3d[1]),
            CrossMamba(dims_3d[0])
        ])

        # ========== 5. U-Net解码器 ==========
        self.decoder = UNetDecoder(channels=dims_3d[::-1])

        # ========== 6. VBP跳跃连接调整 ==========
        self.vbp_jump_adjust = nn.Conv3d(vbp_output_channels, dims_3d[0], kernel_size=1)

        # ========== 7. 最终精炼块 ==========
        self.final_refine = nn.Sequential(
            nn.Conv3d(dims_3d[0] * 2, dims_3d[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(dims_3d[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(dims_3d[0], dims_3d[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(dims_3d[0]),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # ========== 8. 分割头 ==========
        self.seg_head = nn.Sequential(
            nn.Conv3d(dims_3d[0], 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

        # ========== 9. 上采样到256 ==========
        self.upsample_to_256 = nn.Sequential(
            nn.ConvTranspose3d(num_classes, num_classes, kernel_size=2, stride=2),
            nn.ConvTranspose3d(num_classes, num_classes, kernel_size=2, stride=2),
        )

    def forward(self, x_ap, x_lat):
        # 2D编码
        f_ap_2d = self.encoder_2d_ap(x_ap)
        f_lat_2d = self.encoder_2d_lat(x_lat)

        # VBP反投影
        f_ap_3d_base = self.vbp_ap(f_ap_2d, None)
        f_lat_3d_base = self.vbp_lat(f_lat_2d, None)

        f_ap_3d = self.vbp_ap(f_ap_2d, f_lat_3d_base)
        f_lat_3d = self.vbp_lat(f_lat_2d, f_ap_3d_base)

        vbp_jump = f_ap_3d + f_lat_3d

        # 3D编码
        f_ap_3d_multi = self.encoder_3d_ap(f_ap_3d)
        f_lat_3d_multi = self.encoder_3d_lat(f_lat_3d)

        # Cross-Mamba
        cross_outputs = []
        for i in range(self.num_levels):
            cross_feat = self.cross_mambas[i](f_ap_3d_multi[i], f_lat_3d_multi[i])
            cross_outputs.append(cross_feat)

        # U-Net解码
        deepest_feat = cross_outputs[0]
        decoded_feat = self.decoder(deepest_feat, cross_outputs)

        # 与VBP跳跃连接融合
        vbp_jump = self.vbp_jump_adjust(vbp_jump)
        combined = torch.cat([decoded_feat, vbp_jump], dim=1)
        final_feat = self.final_refine(combined)

        # 分割
        seg_out = self.seg_head(final_feat)

        # 上采样
        output = self.upsample_to_256(seg_out)

        return output


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("X2Shape 模型测试 (优化版 - 直接使用InstanceNorm)")
    print("=" * 60)

    model = X2Shape(
        img_size=256,
        in_chans=1,
        num_classes=1,
        dims_2d=[32, 64, 128, 256],
        depths_2d=[1, 1, 2, 1],
        dims_3d=[32, 64, 128, 256],
        vbp_output_channels=64,
        vbp_embed_dim=64
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.2f}M")

    batch_size = 1
    x_ap = torch.randn(batch_size, 1, 256, 256).to(device)
    x_lat = torch.randn(batch_size, 1, 256, 256).to(device)

    print(f"\n前向传播测试...")
    model.eval()
    with torch.no_grad():
        output = model(x_ap, x_lat)

    print(f"输入 AP: {x_ap.shape}")
    print(f"输入 LAT: {x_lat.shape}")
    print(f"输出: {output.shape}")

    expected = (batch_size, 1, 256, 256, 256)
    if output.shape == expected:
        print(f"\n✅ 测试通过！输出为单通道 {output.shape}")
    else:
        print(f"\n❌ 形状错误: 期望 {expected}, 实际 {output.shape}")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")