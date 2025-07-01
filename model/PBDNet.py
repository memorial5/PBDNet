import math
from model.smt import smt_t
from model.MobileNetV2 import mobilenet_v2
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mmcv.cnn import build_norm_layer

TRAIN_SIZE = 384

class PBDNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_backbone = smt_t(pretrained)
        self.d_backbone = mobilenet_v2(pretrained)


        self.conv1 = DSConv3x3(512,256)
        self.conv2 = DSConv3x3(256,128)
        self.conv3 = DSConv3x3(128,64)

        self.fisaf4 = FISAF(512)
        self.fisaf3 = FISAF(256)
        self.fisaf2 = FISAF(128)
        self.fisaf1 = FISAF(64)

#
        self.rspfa1 = RSPFA(512, 256)
        self.rspfa2 = RSPFA(256, 128)
        self.rspfa3 = RSPFA(128, 64)
        # Pred
        # self.mcm3 = SS_MCM(inc=512, outc=256)
        self.msfa3 = MSFA(512, 256)
        self.msfa2 = MSFA(256, 128)
        self.msfa1 = MSFA(128, 64)

        self.d_trans_4 = Trans(320, 512)
        self.d_trans_3 = Trans(96, 256)
        self.d_trans_2 = Trans(32, 128)
        self.d_trans_1 = Trans(24, 64)

        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)
        )


    def forward(self, x_rgb, x_d):
        # rgb
        _, (rgb_1, rgb_2, rgb_3, rgb_4) = self.rgb_backbone(x_rgb)
        # _, rgb_1, rgb_2, rgb_3, rgb_4 = self.d_backbone(x_rgb)
        #
        # rgb_4 = self.d_trans_4(rgb_4)
        # rgb_3 = self.d_trans_3(rgb_3)
        # rgb_2 = self.d_trans_2(rgb_2)
        # rgb_1 = self.d_trans_1(rgb_1)

        # d
        _, d_1, d_2, d_3, d_4 = self.d_backbone(x_d)

        d_4 = self.d_trans_4(d_4)
        d_3 = self.d_trans_3(d_3)
        d_2 = self.d_trans_2(d_2)
        d_1 = self.d_trans_1(d_1)
        #RGB_decoder
        rgb_4_0 = self.rspfa1(rgb_4) # [B, 256, 12, 12]
        rgb_4_0 = F.interpolate(rgb_4_0, scale_factor=2, mode='bilinear', align_corners=True) #[B, 256, 24, 24]
        rgb_3_0 = self.rspfa2(self.conv1(torch.cat((rgb_4_0, rgb_3), dim=1))) #[B, 128, 24, 24]
        rgb_3_0 = F.interpolate(rgb_3_0, scale_factor=2, mode='bilinear', align_corners=True) #[B, 128, 48, 48]
        rgb_2_0 = self.rspfa3(self.conv2(torch.cat((rgb_3_0, rgb_2), dim=1))) # [B, 64, 48, 48]
        rgb_2_0 = F.interpolate(rgb_2_0, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, 96, 96]
        rgb_1_0 = self.conv3(torch.cat((rgb_2_0, rgb_1), dim=1)) # [B, 64, 96, 96]
        pred_r = F.interpolate(self.predtrans1(rgb_1_0), TRAIN_SIZE, mode="bilinear", align_corners=True)
        # depth_decoder
        d_4_0 = self.rspfa1(d_4)  # [B, 256, 12, 12]
        d_4_0 = F.interpolate(d_4_0, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, 24, 24]
        d_3_0 = self.rspfa2(self.conv1(torch.cat((d_4_0, d_3), dim=1)))  # [B, 128, 24, 24]
        d_3_0 = F.interpolate(d_3_0, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 128, 48, 48]
        d_2_0 = self.rspfa3(self.conv2(torch.cat((d_3_0, d_2), dim=1)))  # [B, 64, 48, 48]
        d_2_0 = F.interpolate(d_2_0, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, 96, 96]
        d_1_0 = self.conv3(torch.cat((d_2_0, d_1), dim=1))  # [B, 64, 96, 96]
        pred_d = F.interpolate(self.predtrans1(d_1_0), TRAIN_SIZE, mode="bilinear", align_corners=True)


        # Fuse
        fuse_4 = self.fisaf4(rgb_4, d_4)  # [B, 512, 12, 12]
        fuse_3 = self.fisaf3(rgb_3, d_3)  # [B, 256, 24, 24]
        fuse_2 = self.fisaf2(rgb_2, d_2)  # [B, 128, 48, 48]
        fuse_1 = self.fisaf1(rgb_1, d_1)  # [B, 64, 96, 96]




        # Pred
        pred_4 = F.interpolate(self.predtrans(fuse_4), TRAIN_SIZE, mode="bilinear", align_corners=True)
        pred_4 = pred_4 + pred_d + pred_r

        pred_3, xf_3 = self.msfa3(fuse_4, fuse_3)
        pred_3 = pred_3 + pred_d + pred_r
        pred_2, xf_2 = self.msfa2(xf_3, fuse_2)
        pred_2 = pred_2 + pred_d + pred_r
        pred_1, xf_1 = self.msfa1(xf_2, fuse_1)
        pred_1 = pred_1 + pred_d + pred_r

        return pred_1, pred_2, pred_3, pred_4


class Trans(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.apply(self._init_weights)

    def forward(self, d):
        return self.trans(d)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



#FISAF

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class FISAF(nn.Module):
    def __init__(self, channel):
        super(FISAF, self).__init__()
        self.gate = nn.Sequential(
            DSConv3x3(channel, channel, dilation=8),
            DSConv3x3(channel, channel, dilation=4),
            DSConv3x3(channel, channel, dilation=2),
            nn.Sigmoid()
        )
        self.conv_c1 = DWPWConv(2 * channel, channel)
        self.conv_c2 = DWPWConv(channel, 2)
    #
    def fusion(self, f1, f2, f_vec):
        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2

    def forward(self, DF, SF):
        SA = self.gate(SF)
        DA = self.gate(DF)
        S_D = torch.cat([SA, DA], dim=1)
        S_D = self.conv_c1(S_D)
        S_D = self.conv_c2(S_D)
        F_SA = self.fusion(DA, SA, S_D)

        return F_SA

#MSFA

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        # xa = x * residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MSFA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSFA, self).__init__()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = DSConv3x3(in_ch, out_ch)
        self.aff = AFF(out_ch)
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(in_channels=out_ch, out_channels=1, kernel_size=1)
        )

    def forward(self, fuse_high, fuse_low):
        fuse_high = self.up2(fuse_high)
        fuse_high = self.conv(fuse_high)
        fe_decode = self.aff(fuse_high, fuse_low)
        pred = F.interpolate(self.predtrans(fe_decode), TRAIN_SIZE, mode="bilinear", align_corners=True)
        # fe_decode = fuse_high + fuse_low
        return pred, fe_decode


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#RSPFA

class RSPFA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RSPFA, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )


        self.branch1_0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )

        self.branch2_0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch2_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.branch3_0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch3_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1 = self.branch1_1(x1_0 + x0)

        x2_0 = self.branch2_0(x)
        x2 = self.branch2_1(x2_0 + x1)

        x3_0 = self.branch3_0(x)
        x3 = self.branch3_1(x3_0 + x2)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

