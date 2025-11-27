import torch
import torch.nn as nn

class Identity(nn.Module):
    def forward(self, x): return x

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

def get_activation_layer(activation, param=None):
    if isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "prelu":
            if isinstance(param, int):
                return nn.PReLU(num_parameters=param)
            return nn.PReLU()
        elif activation == "swish":
            return Swish()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "identity":
            return Identity()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
    elif callable(activation):
        try:
            return activation()
        except Exception:
            raise RuntimeError("Callable activation must be an nn.Module class with zero-arg constructor.")
    else:
        assert isinstance(activation, nn.Module)
        return activation

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=False, activation="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = get_activation_layer(activation, out_ch) if activation is not None else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

def conv3x3_block(in_c, out_c, stride=1, activation="relu"):
    return ConvBlock(in_c, out_c, 3, stride=stride, padding=1, activation=activation)

def conv1x1_block(in_c, out_c, stride=1, activation="relu"):
    return ConvBlock(in_c, out_c, 1, stride=stride, padding=0, activation=activation)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4, activation="swish"):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = conv1x1_block(channels, mid, activation=activation)
        self.conv2 = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.conv2(w)
        w = self.gate(w)
        return x * w

def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    if c % groups != 0:
        raise ValueError("channels must be divisible by groups for channel_shuffle")
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x

class MixConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=False, axis=1):
        super().__init__()
        kernel_list = kernel_size if isinstance(kernel_size, (list,tuple)) else [kernel_size]
        pad_list = padding if isinstance(padding, (list,tuple)) else ([padding] * len(kernel_list) if padding is not None else [k//2 for k in kernel_list])
        kcount = len(kernel_list)
        self.splits = self.split_channels(in_channels, kcount)
        out_splits = self.split_channels(out_channels, kcount)
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_list):
            in_c_i = self.splits[i]
            out_c_i = out_splits[i]
            pad_i = pad_list[i]
            conv = nn.Conv2d(in_c_i, out_c_i, kernel_size=k, stride=stride, padding=pad_i, dilation=dilation,
                             groups=(in_c_i if out_c_i == in_c_i and in_c_i>1 else 1), bias=bias)
            self.convs.append(conv)

    def forward(self, x):
        xs = torch.split(x, self.splits, dim=1)
        ys = [conv(xi) for conv, xi in zip(self.convs, xs)]
        return torch.cat(ys, dim=1)

    @staticmethod
    def split_channels(channels, k):
        base = channels // k
        splits = [base] * k
        remainder = channels - base * k
        for i in range(remainder):
            splits[i] += 1
        return splits

class MixConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, activation=(lambda: nn.ReLU(inplace=True))):
        super().__init__()
        self.conv = MixConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation_layer(activation, out_channels) if activation is not None else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class MixUnit(nn.Module):
    def __init__(self, in_c, out_c, stride, exp_kernel_count, conv1_kernel_count, conv2_kernel_count, exp_factor, se_factor, activation, shuffle=True):
        super().__init__()
        self.residual = (in_c == out_c and stride == 1)
        self.shuffle = shuffle
        self.use_se = (se_factor > 0)
        mid = in_c * exp_factor
        if exp_factor > 1:
            if exp_kernel_count == 1:
                self.expand = conv1x1_block(in_c, mid, activation=activation)
            else:
                self.expand = mixconv1x1_block(in_c, mid, kernel_count=exp_kernel_count, activation=activation)
        else:
            self.expand = nn.Identity()
        if conv1_kernel_count == 1:
            self.conv1 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=mid, bias=False)
            self.conv1 = nn.Sequential(self.conv1, nn.BatchNorm2d(mid), get_activation_layer(activation, mid))
        else:
            ks = [3 + 2 * i for i in range(conv1_kernel_count)]
            pads = [1 + i for i in range(conv1_kernel_count)]
            self.conv1 = MixConvBlock(mid, mid, kernel_size=ks, stride=stride, padding=pads, activation=activation)
        if self.use_se:
            self.se = SEBlock(mid, reduction=(exp_factor * se_factor), activation=activation)
        if conv2_kernel_count == 1:
            self.conv2 = conv1x1_block(mid, out_c, activation=None)
        else:
            self.conv2 = mixconv1x1_block(mid, out_c, kernel_count=conv2_kernel_count, activation=None)

    def forward(self, x):
        identity = x if self.residual else None
        x = self.expand(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        if self.shuffle:
            x = channel_shuffle(x, groups=2)
        return x

def mixconv1x1_block(in_channels, out_channels, kernel_count, activation=(lambda: nn.ReLU(inplace=True))):
    return MixConvBlock(in_channels, out_channels, kernel_size=[1]*kernel_count, padding=[0]*kernel_count, activation=activation)

class EmbeddingHead(nn.Module):
    def __init__(self, in_ch, embedding_size=512, embed_expansion=1024, depthwise_kernel=7):
        super().__init__()
        self.embed_expansion = embed_expansion
        self.depthwise_kernel = depthwise_kernel

        self.expand = nn.Conv2d(in_ch, embed_expansion, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(embed_expansion)
        self.expand_act = get_activation_layer("prelu", embed_expansion)

        self.dw = nn.Conv2d(embed_expansion, embed_expansion, kernel_size=depthwise_kernel, stride=1, padding=0,
                            groups=embed_expansion, bias=False)
        self.dw_bn = nn.BatchNorm2d(embed_expansion)

        self.project = nn.Conv2d(embed_expansion, embedding_size, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(embedding_size)

        self.fallback_pool = nn.AdaptiveAvgPool2d(1)
        self.fallback_proj = nn.Conv2d(in_ch, embedding_size, kernel_size=1, bias=False)
        self.fallback_bn = nn.BatchNorm2d(embedding_size)

        self.final_bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        b, c, h, w = x.shape
        if h >= self.depthwise_kernel and w >= self.depthwise_kernel:
            x = self.expand(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)
            x = self.dw(x)
            x = self.dw_bn(x)
            x = self.project(x)
            x = self.project_bn(x)
            x = x.view(b, -1)
            x = self.final_bn(x)
            return x
        else:
            x = self.fallback_pool(x)
            x = self.fallback_proj(x)
            x = self.fallback_bn(x)
            x = x.view(b, -1)
            x = self.final_bn(x)
            return x

class MixFaceNet(nn.Module):
    def __init__(self, cfg, in_channels=3, embedding_size=512, use_shuffle=True, fp16=False, embed_expansion=1024, gdw_size=None):
        super().__init__()
        self.fp16 = fp16
        c_list, k_list, exp_list = cfg["channels"], cfg["kernels"], cfg["exp_factors"]

        self.head = nn.Sequential(
            conv3x3_block(in_channels, c_list[0], stride=2, activation="prelu"),
            MixUnit(c_list[0], c_list[0], stride=1, exp_kernel_count=1, conv1_kernel_count=1, conv2_kernel_count=1,
                    exp_factor=1, se_factor=0, activation="prelu", shuffle=use_shuffle)
        )

        self.stages = nn.Sequential()
        in_c = c_list[0]
        for i, out_c in enumerate(c_list[1:]):
            ks = k_list[i] if i < len(k_list) else [3]
            ef = exp_list[i] if i < len(exp_list) else 1
            stride = 2 if i > 0 else 1
            unit = MixUnit(in_c, out_c, stride=stride,
                           exp_kernel_count=1, conv1_kernel_count=(len(ks) if isinstance(ks, list) else 1),
                           conv2_kernel_count=1, exp_factor=ef, se_factor=4,
                           activation="swish", shuffle=use_shuffle)
            self.stages.add_module(f"stage{i+1}", unit)
            in_c = out_c

        tail_size = gdw_size if gdw_size is not None else in_c
        self.tail = conv1x1_block(in_c, tail_size, activation="prelu") if tail_size != in_c else nn.Identity()

        self.embedding = EmbeddingHead(tail_size, embedding_size=embedding_size, embed_expansion=embed_expansion)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            x = self.head(x)
            x = self.stages(x)
            x = self.tail(x)
            x = self.embedding(x)
        return x.float()

def MixFaceNet_XS(use_shuffle=False, fp16=False, **kwargs):
    cfg = {
        "channels": [8, 12, 16, 32, 64, 128, 256],
        "kernels": [[3], [3,5,7], [3,5,7], [3,5,7], [3,5,7], [3]],
        "exp_factors": [1,1,1,1,1,1],
    }
    return MixFaceNet(cfg, use_shuffle=use_shuffle, fp16=fp16, **kwargs)

def MixFaceNet_S(use_shuffle=False, fp16=False, **kwargs):
    cfg = {
        "channels": [16, 24, 32, 64, 128, 256, 512],
        "kernels": [[3], [3,5,7], [3,5,7], [3,5,7], [3,5,7], [3]],
        "exp_factors": [1,1,1,1,1,1],
    }
    return MixFaceNet(cfg, use_shuffle=use_shuffle, fp16=fp16, **kwargs)

def MixFaceNet_M(use_shuffle=False, fp16=False, **kwargs):
    cfg = {
        "channels": [16, 24, 40, 80, 160, 320, 640],
        "kernels":  [[3], [3,5,7], [3], [3,5,7], [3,5,7], [3]],
        "exp_factors": [1,1,1,1,1,1],
    }
    return MixFaceNet(cfg, use_shuffle=use_shuffle, fp16=fp16, **kwargs)

def ShuffleMixFaceNet_XS(fp16=False, **kwargs):
    return MixFaceNet_XS(use_shuffle=True, fp16=fp16, **kwargs)

def ShuffleMixFaceNet_S(fp16=False, **kwargs):
    return MixFaceNet_S(use_shuffle=True, fp16=fp16, **kwargs)

def ShuffleMixFaceNet_M(fp16=False, **kwargs):
    return MixFaceNet_M(use_shuffle=True, fp16=fp16, **kwargs)