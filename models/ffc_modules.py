# ------------------------------------------------------------------------------
# FFC (Fast Fourier Convolution) modules adapted for PIDNet I-branch integration
# Reference: "Fast Fourier Convolution" (NeurIPS 2020)
# Adapted from: FFC/model_zoo/ffc.py
# Changes:
#   1. Upgraded FourierUnit from deprecated torch.rfft/irfft to torch.fft.rfft2/irfft2
#   2. Added FFCBasicBlock and FFCBottleneck matching PIDNet's no_relu interface
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core FFC building blocks
# ---------------------------------------------------------------------------

class FourierUnit(nn.Module):
    """
    Spectral convolution unit operating in frequency domain.
    Upgraded to use torch.fft.rfft2 / irfft2 (PyTorch >= 1.8).
    """

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        # Real + Imaginary parts are concatenated along channel dim
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0,
            groups=self.groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        batch, c, h, w = x.size()

        # --- FFT ---
        # rfft2 returns complex tensor of shape (batch, c, h, w//2+1)
        ffted = torch.fft.rfft2(x, norm='ortho')                    # complex64

        # Split into real and imaginary, concat along channel axis
        # => (batch, c*2, h, w//2+1)  (real float)
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        # 1×1 conv in frequency domain
        ffted = self.conv_layer(ffted)                               # (batch, out_c*2, h, w//2+1)
        ffted = self.relu(self.bn(ffted))

        # Reconstruct complex tensor
        ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)      # each (batch, out_c, h, w//2+1)
        ffted = torch.complex(ffted_real, ffted_imag)

        # --- IFFT ---
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')   # (batch, out_c, h, w)
        return output


class SpectralTransform(nn.Module):
    """
    Hierarchical spectral transform with optional Local Fourier Unit (LFU).
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=False)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels,
            kernel_size=1, groups=groups, bias=False
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    """
    Fast Fourier Convolution layer.

    Splits channels into local (l) and global (g) parts:
      l: standard spatial convolution
      g: SpectralTransform in frequency domain

    ratio_gin  - fraction of input  channels routed to global branch
    ratio_gout - fraction of output channels produced by global branch
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride in (1, 2), "Stride must be 1 or 2."
        self.stride = stride
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        in_cg  = int(in_channels  * ratio_gin)
        in_cl  = in_channels  - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        # l→l
        self.convl2l = (nn.Identity() if in_cl == 0 or out_cl == 0 else
                        nn.Conv2d(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias))
        # l→g
        self.convl2g = (nn.Identity() if in_cl == 0 or out_cg == 0 else
                        nn.Conv2d(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias))
        # g→l
        self.convg2l = (nn.Identity() if in_cg == 0 or out_cl == 0 else
                        nn.Conv2d(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias))
        # g→g  (SpectralTransform)
        self.convg2g = (nn.Identity() if in_cg == 0 or out_cg == 0 else
                        SpectralTransform(in_cg, out_cg, stride,
                                          1 if groups == 1 else groups // 2, enable_lfu))

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + (self.convg2l(x_g) if not isinstance(x_g, int) else 0)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + (self.convg2g(x_g) if not isinstance(x_g, int) else 0)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    """FFC followed by per-branch BatchNorm + Activation."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)

        out_cl = int(out_channels * (1 - ratio_gout))
        out_cg = int(out_channels * ratio_gout)

        self.bn_l  = nn.Identity() if ratio_gout == 1 else norm_layer(out_cl)
        self.bn_g  = nn.Identity() if ratio_gout == 0 else norm_layer(out_cg)
        # nn.Identity 不接受 inplace 参数，需要安全地实例化激活函数
        def _make_act(act_cls):
            try:
                return act_cls(inplace=False)
            except TypeError:
                return act_cls()

        self.act_l = nn.Identity() if ratio_gout == 1 else _make_act(activation_layer)
        self.act_g = nn.Identity() if ratio_gout == 0 else _make_act(activation_layer)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l)) if not isinstance(x_l, int) else x_l
        x_g = self.act_g(self.bn_g(x_g)) if not isinstance(x_g, int) else x_g
        return x_l, x_g


# ---------------------------------------------------------------------------
# FFC-aware downsample helper (replaces the Conv2d 1×1 shortcut in PIDNet)
# ---------------------------------------------------------------------------

class FFCDownsample(nn.Module):
    """
    Shortcut / downsample projection for FFC residual blocks.
    Handles (x_l, x_g) tuples; produces matching (id_l, id_g).
    """

    def __init__(self, in_channels, out_channels, stride,
                 ratio_gin, ratio_gout, enable_lfu=True):
        super(FFCDownsample, self).__init__()
        self.ffc_bn_act = FFC_BN_ACT(
            in_channels, out_channels,
            kernel_size=1, stride=stride,
            ratio_gin=ratio_gin, ratio_gout=ratio_gout,
            enable_lfu=enable_lfu
        )

    def forward(self, x):
        return self.ffc_bn_act(x)


# ---------------------------------------------------------------------------
# FFC BasicBlock — drop-in replacement for PIDNet's BasicBlock in I-branch
# ---------------------------------------------------------------------------

class FFCBasicBlock(nn.Module):
    """
    FFC-aware BasicBlock matching PIDNet's BasicBlock interface.

    Args:
        inplanes   : input channels (total, before split)
        planes     : output channels (total)
        stride     : spatial stride
        downsample : None or FFCDownsample instance
        ratio_gin  : fraction of *input* channels for global branch
        ratio_gout : fraction of *output* channels for global branch
        enable_lfu : enable Local Fourier Unit inside SpectralTransform
        no_relu    : if True, do not apply ReLU after residual add (matches PIDNet)

    Input / Output: tuple (x_l, x_g)  —  local and global feature maps
    When ratio_gin == 0, input can be a plain tensor x (treated as (x, 0)).
    When ratio_gout == 0, output is still a tuple (x_l, 0); caller merges with x_l.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 ratio_gin=0.5, ratio_gout=0.5,
                 enable_lfu=True, no_relu=False):
        super(FFCBasicBlock, self).__init__()

        self.conv1 = FFC_BN_ACT(
            inplanes, planes,
            kernel_size=3, padding=1, stride=stride,
            ratio_gin=ratio_gin, ratio_gout=ratio_gout,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            enable_lfu=enable_lfu
        )
        self.conv2 = FFC_BN_ACT(
            planes, planes * self.expansion,
            kernel_size=3, padding=1,
            ratio_gin=ratio_gout, ratio_gout=ratio_gout,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.Identity,   # ReLU applied after residual add
            enable_lfu=enable_lfu
        )

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        # Per-branch post-add activation
        self.relu_l = nn.Identity() if ratio_gout == 1   else nn.ReLU(inplace=False)
        self.relu_g = nn.Identity() if ratio_gout == 0   else nn.ReLU(inplace=False)

        self.ratio_gin  = ratio_gin
        self.ratio_gout = ratio_gout

    def forward(self, x):
        # Accept plain tensor for the entry point (ratio_gin == 0)
        x = x if isinstance(x, tuple) else (x, 0)

        if self.downsample is not None:
            id_l, id_g = self.downsample(x)
        else:
            id_l, id_g = x

        out = self.conv1(x)
        out_l, out_g = self.conv2(out)

        # Residual add per branch
        out_l = out_l + id_l if not isinstance(out_l, int) else out_l
        out_g = out_g + id_g if not isinstance(out_g, int) else out_g

        if not self.no_relu:
            out_l = self.relu_l(out_l) if not isinstance(out_l, int) else out_l
            out_g = self.relu_g(out_g) if not isinstance(out_g, int) else out_g

        return out_l, out_g


# ---------------------------------------------------------------------------
# FFC Bottleneck — for layer5 in PIDNet (expansion=2, ratio_gout defaults to 0)
# ---------------------------------------------------------------------------

class FFCBottleneck(nn.Module):
    """
    FFC-aware Bottleneck matching PIDNet's Bottleneck interface.

    Set ratio_gout=0 at the exit layer to collapse the global branch back
    into a standard tensor before DAPPM / PAPPM.

    Input / Output: tuple (x_l, x_g)
    """
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 ratio_gin=0.5, ratio_gout=0.0,
                 enable_lfu=True, no_relu=True):
        super(FFCBottleneck, self).__init__()

        self.conv1 = FFC_BN_ACT(
            inplanes, planes,
            kernel_size=1,
            ratio_gin=ratio_gin, ratio_gout=ratio_gout,
            activation_layer=nn.ReLU,
            enable_lfu=enable_lfu
        )
        self.conv2 = FFC_BN_ACT(
            planes, planes,
            kernel_size=3, padding=1, stride=stride,
            ratio_gin=ratio_gout, ratio_gout=ratio_gout,
            activation_layer=nn.ReLU,
            enable_lfu=enable_lfu
        )
        self.conv3 = FFC_BN_ACT(
            planes, planes * self.expansion,
            kernel_size=1,
            ratio_gin=ratio_gout, ratio_gout=ratio_gout,
            activation_layer=nn.Identity,
            enable_lfu=enable_lfu
        )

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=False)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=False)

        self.ratio_gin  = ratio_gin
        self.ratio_gout = ratio_gout

    def forward(self, x):
        x = x if isinstance(x, tuple) else (x, 0)

        if self.downsample is not None:
            id_l, id_g = self.downsample(x)
        else:
            id_l, id_g = x

        out = self.conv1(x)
        out = self.conv2(out)
        out_l, out_g = self.conv3(out)

        out_l = out_l + id_l if not isinstance(out_l, int) else out_l
        out_g = out_g + id_g if not isinstance(out_g, int) else out_g

        if not self.no_relu:
            out_l = self.relu_l(out_l) if not isinstance(out_l, int) else out_l
            out_g = self.relu_g(out_g) if not isinstance(out_g, int) else out_g

        return out_l, out_g
