from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


class conv(nn.Module):
    """_summary_
        自己又把卷积层定义了一遍，就是卷积+batchnorm+relu
    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        # 卷积的输入通道数、输出通道数、卷积核大小、stride步长。其实这里使用padding也可以，但是这里没有使用，在下面使用了
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        # 输出图像是 （输入宽高 - kernel_size + 2*padding）/ stride + 1
        # 假设kernel_size=7,stride=2, 输入是224*224。下面算的p = 3, 230*230, 115+1 = 116  ???
        p = int(np.floor((self.kernel_size-1)/2))  # 向下取整
        p2d = (p, p, p, p)  # 计算图像上下左右需要pad的大小
        x = self.conv_base(F.pad(x, p2d))  # 先用F.pad对图像进行padding，然后送入卷积层
        x = self.normalize(x)  # batchnorm
        return F.elu(x, inplace=True)  # 激活函数，elu是另一种激活函数


class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        # 注意这是自定义的卷积层，里面包括了卷积+BN+relu
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)  
        x_out = self.conv2(x_out)   # 这里已经有一个relu了， 下面加起来又过了一个relu，有点不对？
        if do_proj:
            shortcut = self.conv3(x)  # 1x1卷积，为了相加
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    """_summary_
        残差块
    """
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)   # 这里*就是把列表中的元素都取出来，然后当做参数传入函数


class upconv(nn.Module):
    """_summary_
        上采样卷积，把图像放大
    """
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)  # 先卷积一次

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        # 输入128通道，输出2通道？
        # 解答：输出2通道就是左视差图和右视差图！
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)  # BN
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        # ! 疑问：这里使用sigmoid转到0-1之间，因为视差不可能是负数。但是这里为什么要*0.3呢？
        return 0.3 * self.sigmoid(x)  # 这是啥啊？为啥要sigmod?


class Resnet50_md(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet50_md, self).__init__()
        # encoder
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock(64, 64, 3, 2)  # H/8  -  256D
        self.conv3 = resblock(256, 128, 4, 2)  # H/16 -  512D
        self.conv4 = resblock(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5 = resblock(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.upconv6 = upconv(2048, 512, 3, 2)
        self.iconv6 = conv(1024 + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(256+128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64+2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(32+64+2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4


class Resnet18_md(nn.Module):
    """_summary_
        定义整个网络架构
    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self, num_in_layers):
        super(Resnet18_md, self).__init__()
        # Step 1 encoder
        # num_in_layers是输入通道数，其实就是3通道rgb图像
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        # Step 2 decoder
        self.upconv6 = upconv(512, 512, 3, 2)   # 卷积上采样，放大两倍
        self.iconv6 = conv(256+512, 512, 3, 1)  

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(128+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)   
        self.iconv4 = conv(64+128, 128, 3, 1)   # 输出为 /8
        self.disp4_layer = get_disp(128)  # 输出/8分辨率下预测的两个视差图

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)   # 输出/4分辨率下预测的两个视差图

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+32 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)   # 输出/2分辨率下预测的两个视差图

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)  # 输出原始分辨率下预测的两个视差图

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)   # /2
        x_pool1 = self.pool1(x1)  # /4
        x2 = self.conv2(x_pool1)  # /8
        x3 = self.conv3(x2)  # /16
        x4 = self.conv4(x3)  # /32
        x5 = self.conv5(x4)  # /64

        # skips
        skip1 = x1       # /2
        skip2 = x_pool1  # /4
        skip3 = x2       # /8
        skip4 = x3       # /16
        skip5 = x4       # /32

        # decoder
        upconv6 = self.upconv6(x5)   # 上采样，/64 -> /32
        concat6 = torch.cat((upconv6, skip5), 1)   # 跳层链接输入和输出
        iconv6 = self.iconv6(concat6) # 大小不变，还是 /32

        upconv5 = self.upconv5(iconv6)  # 上采样，/32 -> /16
        concat5 = torch.cat((upconv5, skip4), 1)  # 跳层链接输入和输出
        iconv5 = self.iconv5(concat5) # 大小不变，还是 /16

        upconv4 = self.upconv4(iconv5)  # 上采样，/16 -> /8
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)   # 大小不变，还是 /8
        self.disp4 = self.disp4_layer(iconv4)  # 输出/8分辨率下预测的两个视差图
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)  # /4
        # ! 注意这里的堆叠：把输入/4分辨率、输出/8上采样到/4分辨率、以及预测的/8分辨率的视差上采样到/4分辨率都堆叠起来
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)  # 1表示按照通道维进行堆叠
        iconv3 = self.iconv3(concat3)   # 正常卷积融合上面堆叠的三个通道，然后分辨率不变
        self.disp3 = self.disp3_layer(iconv3)   # 输出/4分辨率下预测的两个视差图
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)  
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)  # 输出/2分辨率下预测的两个视差图
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)  # 输出原图分辨率下预测的两个视差图
        return self.disp1, self.disp2, self.disp3, self.disp4  # 返回，也就是输出的就是4个分辨率下预测的深度图


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class ResnetModel(nn.Module):
    def __init__(self, num_in_layers, encoder='resnet18', pretrained=False):
        super(ResnetModel, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50',\
                           'resnet101', 'resnet152'],\
                           "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)\
                                (pretrained=pretrained)
        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(num_in_layers, 64,
                              kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
        else:
            self.firstconv = resnet.conv1 # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool # H/4

        # encoder
        self.encoder1 = resnet.layer1 # H/4
        self.encoder2 = resnet.layer2 # H/8
        self.encoder3 = resnet.layer3 # H/16
        self.encoder4 = resnet.layer4 # H/32

        # decoder
        self.upconv6 = upconv(filters[3], 512, 3, 2)
        self.iconv6 = conv(filters[2] + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(filters[1] + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(filters[0] + 128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 1) #
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + 2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x_first_conv = self.firstconv(x)
        x = self.firstbn(x_first_conv)
        x = self.firstrelu(x)
        x_pool1 = self.firstmaxpool(x)
        x1 = self.encoder1(x_pool1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        # skips
        skip1 = x_first_conv
        skip2 = x_pool1
        skip3 = x1
        skip4 = x2
        skip5 = x3

        # decoder
        upconv6 = self.upconv6(x4)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=1, mode='bilinear', align_corners=True)
        self.disp4 = nn.functional.interpolate(self.disp4, scale_factor=0.5, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4
