import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


class NonLocalBlock(nn.Module):
  def __init__(self, in_channels, shape):
    super(NonLocalBlock, self).__init__()

    self.in_channels = in_channels
    self.inter_channels = in_channels // 8
    inter_channels = in_channels // 8
    conv_nd = nn.Conv2d
    self.bn = BatchNorm2d(in_channels, momentum=bn_mom)

    self.w_v = nn.Sequential(
      conv_nd(in_channels=in_channels, out_channels=in_channels,
              kernel_size=1, stride=1, padding=0, bias=False)
    )

    self.w_q = nn.Sequential(
      conv_nd(in_channels=in_channels, out_channels=inter_channels,
              kernel_size=1, stride=1, padding=0, bias=False)
    )

    self.w_k = nn.Sequential(
      conv_nd(in_channels=in_channels, out_channels=inter_channels,
              kernel_size=1, stride=1, padding=0, bias=False),
    )

    self.distance = utils.gen_distance(shape).cuda()
    self.relu = nn.ReLU()

  def forward(self, x):
    batch_size = x.size(0)

    v = self.w_v(x).view(batch_size, self.in_channels, -1)
    v = v.permute(0, 2, 1)

    q = self.w_q(x).view(batch_size, self.inter_channels, -1)
    q = q.permute(0, 2, 1)
    k = self.w_k(x).view(batch_size, self.inter_channels, -1)

    relation = torch.matmul(q, k)
    relation = relation / self.distance
    relation = F.softmax(relation, dim=-1)

    y = torch.matmul(relation, v)
    y = y.permute(0, 2, 1).contiguous()
    y = y.view(batch_size, self.in_channels, *x.size()[2:])

    return self.relu(self.bn(y)) + x


class ConvBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, input):
    x = self.conv1(input)
    return self.relu(self.bn(x))


class AttentionRefinementModule(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

  def forward(self, input):
    x = self.avgpool(input)
    assert self.in_channels == x.size(1)
    x = self.conv(x)
    x = self.bn(x)
    x = self.sigmoid(x)
    x = torch.mul(input, x)

    x = self.bn(x)
    x = self.relu(x)
    return x


class FeatureFusionModule(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=self.out_channels, stride=1)
    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    self.sigmoid = nn.Sigmoid()
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

  def forward(self, input_1, input_2):
    x = torch.cat((input_1, input_2), dim=1)
    assert self.in_channels == x.size(1)
    feature = self.convblock(x)

    x = self.avgpool(feature)
    x = self.relu(self.bn(self.conv1(x)))
    x = self.sigmoid(self.bn(self.conv2(x)))
    x = torch.mul(feature, x)

    x = torch.add(x, feature)

    x = self.bn(x)
    x = self.relu(x)
    return x


class EEB(nn.Module):
  def __init__(self, in_channels, out_channels, inter_scale=4):
    super(EEB, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, int(in_channels / inter_scale), kernel_size=1, stride=1, padding=0)

    self.conv2 = nn.Conv2d(int(in_channels / inter_scale), int(in_channels / inter_scale), kernel_size=3, stride=1,
                           padding=1)
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm2d(int(in_channels / inter_scale))
    self.conv3 = nn.Conv2d(int(in_channels / inter_scale), int(in_channels / inter_scale), kernel_size=3, stride=1,
                           padding=1)

    self.conv4 = nn.Conv2d(int(in_channels / inter_scale), out_channels, kernel_size=1, stride=1, padding=0)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x, relu=True):
    x = self.conv1(x)

    res = self.conv2(x)
    res = self.bn(res)
    res = self.relu(res)
    res = self.conv3(res)
    res = self.bn(res)
    res = self.relu(res)

    x = self.conv4(x + res)
    if relu:
      return self.relu(self.bn2(x))
    else:
      return self.bn2(x)


class BasicBlock(nn.Module):
  expansion = 1
  __constants__ = ['downsample']

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3l = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4l = self._make_layer(block, 512, layers[3], stride=2)  # different

    self.cons_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, bias=False, padding=2)

    self.inplanes = 128
    self.layer3h = self._make_layer(block, 256, layers[2], stride=1)
    self.layer4h = self._make_layer(block, 512, layers[3], stride=1)

    self.eeb1 = EEB(64, 1)
    self.eeb2 = EEB(128, 1)
    self.eeb3h = EEB(256, 1)
    self.eeb4h = EEB(512, 1)
    self.eeb3l = EEB(256, 1)
    self.eeb4l = EEB(512, 1)

    self.eeb_merge = nn.Sequential(nn.Conv2d(6, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(1))

    self.arm1 = AttentionRefinementModule(256, 256)
    self.arm2 = AttentionRefinementModule(512, 512)
    self.ffm = FeatureFusionModule(256 + 512, 256)

    self.non_local1 = NonLocalBlock(256, 32)
    self.non_local2 = NonLocalBlock(512, 16)
    self.down_channel4h = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)


    self.down_channel1 = nn.Sequential(
      nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(64),
      self.relu
    )
    self.down_channel2 = nn.Sequential(
      nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(16),
      self.relu
    )
    self.down_channel3 = nn.Sequential(
      nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(1)
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    with torch.no_grad():
      self.cons_conv.weight.copy_(utils.gen_cons_conv_weight(5))

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.cons_conv(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    edge1 = self.eeb1(x)

    x = self.layer2(x)
    edge2 = self.eeb2(x)
    edge2 = F.interpolate(edge2, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)

    xh = self.layer3h(x)
    xl = self.layer3l(x)
    xl_arm1 = self.arm1(xl)
    non_local1 = self.non_local1(xl_arm1)
    non_local1 = F.interpolate(non_local1, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)

    edge3h = self.eeb3h(xh)
    edge3h = F.interpolate(edge3h, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    edge3l = self.eeb3l(xl)
    edge3l = F.interpolate(edge3l, scale_factor=4, mode='bilinear', align_corners=Config.align_corners)

    xh = self.layer4h(xh)
    xl = self.layer4l(xl)
    xl_arm2 = self.arm2(xl)
    non_local2 = self.non_local2(xl_arm2)
    non_local2 = F.interpolate(non_local2, scale_factor=4, mode='bilinear', align_corners=Config.align_corners)

    edge4h = self.eeb4h(xh)
    edge4h = F.interpolate(edge4h, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    edge4l = self.eeb4l(xl)
    edge4l = F.interpolate(edge4l, scale_factor=8, mode='bilinear', align_corners=Config.align_corners)

    edge = self.eeb_merge(torch.cat((edge1, edge2, edge3h, edge3l, edge4h, edge4l), dim=1))

    x_merge = self.ffm(xh + non_local2, self.down_channel4h(xh) + non_local1)


    x = F.interpolate(x_merge, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel1(x)

    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel2(x)

    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel3(x)

    return x, edge


def get_seg_model():
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  return model

