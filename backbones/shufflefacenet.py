import torch
from torch import nn

__all__ = ['shufflefacenet_05x', 'shufflefacenet_1x', 'shufflefacenet_15x', 'shufflefacenet_2x', 'get_shufflefacenet']

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if stride == 1 and in_channels == out_channels:
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.PReLU(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.PReLU(mid_channels)
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.PReLU(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels - in_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels - in_channels),
                nn.PReLU(out_channels - in_channels)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.PReLU(in_channels)
            )

    def forward(self, x):
        if hasattr(self, 'branch1'):
            out = torch.cat([self.branch1(x), self.branch2(x)], 1)
        else:
            c = x.shape[1] // 2
            out = torch.cat([x[:, :c], self.branch2(x[:, c:])], 1)
        return channel_shuffle(out, 2)

class ShuffleFaceNet(nn.Module):
    def __init__(self, scale=1.0, num_features=512, fp16=False, dropout=0.0):
        super().__init__()
        self.fp16 = fp16
        self.num_features = num_features

        channels_dict = {
            0.5: [24, 48, 96, 192],
            1.0: [24, 116, 232, 464],
            1.5: [24, 176, 352, 704],
            2.0: [24, 244, 488, 976]
        }
        stage_out = channels_dict[scale]

        conv5_out = 1024 if scale < 2.0 else 2048

        stage_repeats = [4, 8, 4]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, stage_out[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(stage_out[0]),
            nn.PReLU(stage_out[0])
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.stage2 = self._make_stage(stage_out[0], stage_out[1], stage_repeats[0])
        self.stage3 = self._make_stage(stage_out[1], stage_out[2], stage_repeats[1])
        self.stage4 = self._make_stage(stage_out[2], stage_out[3], stage_repeats[2], first_stride=1)

        self.conv5 = nn.Sequential(
            nn.Conv2d(stage_out[3], conv5_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(conv5_out),
            nn.PReLU(conv5_out)
        )

        self.gdc = nn.Sequential(
            nn.Conv2d(conv5_out, conv5_out, 7, 1, 0, groups=conv5_out, bias=False),
            nn.BatchNorm2d(conv5_out),
            nn.PReLU(conv5_out)
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.fc = nn.Linear(conv5_out, num_features)
        self.bn = nn.BatchNorm1d(num_features)

        self._initialize_weights()

    def _make_stage(self, in_c, out_c, repeat, first_stride=2):
        layers = [ShuffleUnit(in_c, out_c, stride=first_stride)]
        for _ in range(1, repeat):
            layers.append(ShuffleUnit(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = self.gdc(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.bn(x)
        return x

def shufflefacenet_05x(**kw): return ShuffleFaceNet(scale=0.5, **kw)
def shufflefacenet_1x(**kw):  return ShuffleFaceNet(scale=1.0, **kw)
def shufflefacenet_15x(**kw): return ShuffleFaceNet(scale=1.5, **kw)
def shufflefacenet_2x(**kw):  return ShuffleFaceNet(scale=2.0, **kw)
