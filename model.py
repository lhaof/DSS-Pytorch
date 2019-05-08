import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)

class DSSNet(nn.Module):

    def __init__(self):
        super(DSSNet, self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=5,dilation=1,bias=True), # conv1_1
        nn.ReLU(inplace=True), 
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv1_2
        nn.ReLU(inplace=True) 
        )
        self.conv2 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool1
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv2_1
        nn.ReLU(inplace=True), 
        nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv2_2
        nn.ReLU(inplace=True) 
        )
        self.conv3 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool3
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv3_1
        nn.ReLU(inplace=True), 
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv3_2
        nn.ReLU(inplace=True), 
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv3_3
        nn.ReLU(inplace=True) 
        )
        self.conv4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool3
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv4_1 
        nn.ReLU(inplace=True), 
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv4_2
        nn.ReLU(inplace=True), 
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv4_3
        nn.ReLU(inplace=True) 
        )
        self.conv5 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True), # pool4
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv5_1 
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv5_2
        nn.ReLU(inplace=True), 
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,bias=True), # conv5_3
        nn.ReLU(inplace=True) 
        )
        self.pool5_twice = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True), # pool5
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True) # pool5_a
        )

        ### dsn 6
        self.conv1_dsn6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=7,padding=3)
        self.relu1_dsn6 = nn.ReLU(inplace=True)
        self.conv2_dsn6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=7,padding=3)
        self.relu2_dsn6 = nn.ReLU(inplace=True)
        self.conv3_dsn6 = nn.Conv2d(in_channels=512,out_channels=1,kernel_size=1,padding=0)
        self.upsample32_in_dsn6 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=64,stride=32,padding=0, bias=True)

        ### dsn 5
        self.conv1_dsn5 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=5,padding=2)
        self.relu1_dsn5 = nn.ReLU(inplace=True)
        self.conv2_dsn5 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=5,padding=2)
        self.relu2_dsn5 = nn.ReLU(inplace=True)
        self.conv3_dsn5 = nn.Conv2d(in_channels=512,out_channels=1,kernel_size=1,padding=0)
        self.upsample16_in_dsn5 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=32,stride=16,padding=0, bias=True)

        ### dsn 4
        self.conv1_dsn4 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=5,padding=2)
        self.relu1_dsn4 = nn.ReLU(inplace=True)
        self.conv2_dsn4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,padding=2)
        self.relu2_dsn4 = nn.ReLU(inplace=True)
        self.conv3_dsn4 = nn.Conv2d(in_channels=256,out_channels=1,kernel_size=1,padding=0)
        self.conv4_dsn4 = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1,padding=0)
        self.upsample4_dsn6 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=8,stride=4,padding=0, bias=True)
        self.upsample2_dsn5 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2,padding=0, bias=True)
        self.upsample8_in_dsn4 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=16,stride=8,padding=0, bias=True)

        ### dsn 3
        self.conv1_dsn3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,padding=2)
        self.relu1_dsn3 = nn.ReLU(inplace=True)
        self.conv2_dsn3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,padding=2)
        self.relu2_dsn3 = nn.ReLU(inplace=True)
        self.conv3_dsn3 = nn.Conv2d(in_channels=256,out_channels=1,kernel_size=1,padding=0)
        self.conv4_dsn3 = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1,padding=0)
        self.upsample8_dsn6 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=16,stride=8,padding=0, bias=True)
        self.upsample4_dsn5 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=8,stride=4,padding=0, bias=True)
        self.upsample4_in_dsn3 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=8,stride=4,padding=0, bias=True)

        ### dsn 2
        self.conv1_dsn2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.relu1_dsn2 = nn.ReLU(inplace=True)
        self.conv2_dsn2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.relu2_dsn2 = nn.ReLU(inplace=True)
        self.conv3_dsn2 = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1,padding=0)
        self.conv4_dsn2 = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1,padding=0)
        self.upsample16_dsn6 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=32,stride=16,padding=0,bias=True)
        self.upsample8_dsn5 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=16,stride=8,padding=0, bias=True)
        self.upsample4_dsn4 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=8,stride=4,padding=0, bias=True)
        self.upsample2_dsn3 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2,padding=0, bias=True)
        self.upsample2_in_dsn2 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2,padding=0, bias=True)

        ### dsn 1
        self.conv1_dsn1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.relu1_dsn1 = nn.ReLU(inplace=True)
        self.conv2_dsn1 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.relu2_dsn1 = nn.ReLU(inplace=True)
        self.conv3_dsn1 = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1,padding=0)
        self.conv4_dsn1 = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1,padding=0)
        self.upsample32_dsn6 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=64,stride=32,padding=0,
            bias=True)
        self.upsample16_dsn5 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=32,stride=16,padding=0,
            bias=True)
        self.upsample8_dsn4 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=16,stride=8,padding=0,
            bias=True)
        self.upsample4_dsn3 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=8,stride=4,padding=0,
            bias=True)

        self.conv_fuse = nn.Conv2d(in_channels=6,out_channels=1,kernel_size=1,stride=1,padding=0,dilation=1,bias=True)

        self._init_weight()

        ## init deconvolution as bilinear interpolation
        params_key = list(self.state_dict().keys())
        for key in params_key:
            if key.startswith('upsample'):
                print('key: ',key)
                if key.endswith('weight'):
                    m, k, h, w = self.state_dict()[key].size()
                    if m != k:
                        print 'input and output channels need to be the same'
                        raise
                    if h != w:
                        print 'filters need to be square'
                        raise
                    filt = upsample_filt(h)
                    filt = torch.tensor(filt.astype(np.float32))
                    assert(self.state_dict()[key].data.size()[2:] == filt.size())
                    assert(m == 1 and k == 1)
                    self.state_dict()[key].data.copy_(filt)
                elif key.endswith('bias'):
                    self.state_dict()[key].zero_()
            elif 'conv_fuse' in key:
                if key.endswith('weight'):
                    m, k, h, w = self.state_dict()[key].size()
                    # m is out_channels, k is in_channels
                    assert(m == 1 and k > 0 and h == 1 and w == 1)
                    self.state_dict()[key].data.fill_(1.0/k)
                elif key.endswith('bias'):
                    self.state_dict()[key].zero_()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def crop(self, x, dsn):
        height, width = x.shape[-2], x.shape[-1]
        return dsn[:, :, :height, :width]

    def get_feats(self, data):
        feat_relu1_2 = self.conv1(data)
        feat_relu2_2 = self.conv2(feat_relu1_2)
        feat_relu3_3 = self.conv3(feat_relu2_2)
        feat_relu4_3 = self.conv4(feat_relu3_3)
        feat_relu5_3 = self.conv5(feat_relu4_3)
        feat_pool5 = self.pool5_twice(feat_relu5_3)
        return feat_relu1_2, feat_relu2_2, feat_relu3_3, feat_relu4_3, feat_relu5_3, feat_pool5

    def forward(self, data):
        relu1_2, relu2_2, relu3_3, relu4_3, relu5_3, pool5 = self.get_feats(data)
        # dsn 6
        conv3_dsn6_feat = self.conv3_dsn6(self.relu2_dsn6(self.conv2_dsn6(self.relu1_dsn6(self.conv1_dsn6(pool5)))))
        upsample32_in_dsn6_feat = self.upsample32_in_dsn6(conv3_dsn6_feat)
        crop_score_dsn6_up = self.crop(data, upsample32_in_dsn6_feat)

        # dsn 5
        conv3_dsn5_feat = self.conv3_dsn5(self.relu2_dsn5(self.conv2_dsn5(self.relu1_dsn5(self.conv1_dsn5(relu5_3)))))
        upsample16_in_dsn5_feat = self.upsample16_in_dsn5(conv3_dsn5_feat)
        crop_score_dsn5_up = self.crop(data, upsample16_in_dsn5_feat)

        # dsn 4
        conv3_dsn4_feat = self.conv3_dsn4(self.relu2_dsn4(self.conv2_dsn4(self.relu1_dsn4(self.conv1_dsn4(relu4_3)))))
        upsample4_dsn6_feat = self.upsample4_dsn6(conv3_dsn6_feat)
        upsample2_dsn5_feat = self.upsample2_dsn5(conv3_dsn5_feat)
        crop_score_dsn6_up_4 = self.crop(conv3_dsn4_feat, upsample4_dsn6_feat)
        crop_score_dsn5_up_4 = self.crop(conv3_dsn4_feat, upsample2_dsn5_feat)
        concat_dsn4 = torch.cat([conv3_dsn4_feat, crop_score_dsn6_up_4, crop_score_dsn5_up_4], dim=1)
        conv4_dsn4_feat = self.conv4_dsn4(concat_dsn4)
        upsample8_in_dsn4_feat = self.upsample8_in_dsn4(conv4_dsn4_feat)
        crop_score_dsn4_up = self.crop(data, upsample8_in_dsn4_feat)


        # dsn 3
        conv3_dsn3_feat = self.conv3_dsn3(self.relu2_dsn3(self.conv2_dsn3(self.relu1_dsn3(self.conv1_dsn3(relu3_3)))))
        upsample8_dsn6_feat = self.upsample8_dsn6(conv3_dsn6_feat)
        upsample4_dsn5_feat = self.upsample4_dsn5(conv3_dsn5_feat)
        crop_score_dsn6_up_3 = self.crop(conv3_dsn3_feat, upsample8_dsn6_feat)
        crop_score_dsn5_up_3 = self.crop(conv3_dsn3_feat, upsample4_dsn5_feat)
        concat_dsn3 = torch.cat([conv3_dsn3_feat, crop_score_dsn6_up_3, crop_score_dsn5_up_3], dim=1)
        conv4_dsn3_feat = self.conv4_dsn3(concat_dsn3)
        upsample4_in_dsn3_feat = self.upsample4_in_dsn3(conv4_dsn3_feat)
        crop_score_dsn3_up = self.crop(data, upsample4_in_dsn3_feat)


        # dsn 2
        conv3_dsn2_feat = self.conv3_dsn2(self.relu2_dsn2(self.conv2_dsn2(self.relu1_dsn2(self.conv1_dsn2(relu2_2)))))
        upsample16_dsn6_feat = self.upsample16_dsn6(conv3_dsn6_feat)
        upsample8_dsn5_feat = self.upsample8_dsn5(conv3_dsn5_feat)
        upsample4_dsn4_feat = self.upsample4_dsn4(conv4_dsn4_feat)
        upsample2_dsn3_feat = self.upsample2_dsn3(conv4_dsn3_feat)
        crop_score_dsn6_up_2 = self.crop(conv3_dsn2_feat, upsample16_dsn6_feat)
        crop_score_dsn5_up_2 = self.crop(conv3_dsn2_feat, upsample8_dsn5_feat)
        crop_score_dsn4_up_2 = self.crop(conv3_dsn2_feat, upsample4_dsn4_feat)
        crop_score_dsn3_up_2 = self.crop(conv3_dsn2_feat, upsample2_dsn3_feat)
        concat_dsn2 = torch.cat((conv3_dsn2_feat, crop_score_dsn5_up_2, crop_score_dsn4_up_2, crop_score_dsn6_up_2, crop_score_dsn3_up_2), dim=1)
        conv4_dsn2_feat = self.conv4_dsn2(concat_dsn2)
        upsample2_in_dsn2_feat = self.upsample2_in_dsn2(conv4_dsn2_feat)
        crop_score_dsn2_up = self.crop(data, upsample2_in_dsn2_feat)


        # dsn 1
        conv3_dsn1_feat = self.conv3_dsn1(self.relu2_dsn1(self.conv2_dsn1(self.relu1_dsn1(self.conv1_dsn1(relu1_2)))))
        upsample32_dsn6_feat = self.upsample32_dsn6(conv3_dsn6_feat)
        upsample16_dsn5_feat = self.upsample16_dsn5(conv3_dsn5_feat)
        upsample8_dsn4_feat = self.upsample8_dsn4(conv4_dsn4_feat)
        upsample4_dsn3_feat = self.upsample4_dsn3(conv4_dsn3_feat)
        crop_score_dsn6_up_1 = self.crop(conv3_dsn1_feat, upsample32_dsn6_feat)
        crop_score_dsn5_up_1 = self.crop(conv3_dsn1_feat, upsample16_dsn5_feat)
        crop_score_dsn4_up_1 = self.crop(conv3_dsn1_feat, upsample8_dsn4_feat)
        crop_score_dsn3_up_1 = self.crop(conv3_dsn1_feat, upsample4_dsn3_feat)
        concat_dsn1 = torch.cat((conv3_dsn1_feat, crop_score_dsn5_up_1, crop_score_dsn4_up_1, crop_score_dsn6_up_1, crop_score_dsn3_up_1), dim=1)
        conv4_dsn1_feat = self.conv4_dsn1(concat_dsn1)
        crop_score_dsn1_up = self.crop(data, conv4_dsn1_feat)

        # fuse
        concat = torch.cat((crop_score_dsn1_up, crop_score_dsn2_up, crop_score_dsn3_up, crop_score_dsn4_up, crop_score_dsn5_up, crop_score_dsn6_up), dim=1)
        new_score_weighting = self.conv_fuse(concat)
        return crop_score_dsn1_up, crop_score_dsn2_up, crop_score_dsn3_up, crop_score_dsn4_up, crop_score_dsn5_up, crop_score_dsn6_up, new_score_weighting

def get_params(model, lr):
    # params_dict = model.state_dict(), different from the following line
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'upsample' not in key:
            if ('conv3_dsn6' in key) or ('conv3_dsn5' in key) or ('conv3_dsn4' in key) or (
                'conv4_dsn4' in key) or ('conv3_dsn3' in key) or ('conv4_dsn3' in key) or (
                'conv3_dsn2' in key) or ('conv4_dsn2' in key) or ('conv3_dsn1' in key) or (
                'conv4_dsn1' in key) or ('conv_fuse' in key):
                if key.endswith('weight'):
                    params += [{'params':[value], 'lr':lr*0.1}]
                elif key.endswith('bias'):
                    params += [{'params':[value], 'lr':lr*0.2}]
                else:
                    print('unknown key: ',key)
                    raise NotImplementedError
            else:
                if key.endswith('weight'):
                    params += [{'params':[value], 'lr':lr*1.0}]
                elif key.endswith('bias'):
                    params += [{'params':[value], 'lr':lr*2.0}]
                else:
                    print('unknown key: ',key)
                    raise NotImplementedError
    return params

class dssloss(nn.Module):
    def __init__(self, weight=[1.0] * 7):
        super(dssloss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        n, c, h, w = x_list[0].size()
        loss = self.weight[0] * nn.BCEWithLogitsLoss(weight=None, size_average=False)(x_list[0], label)
        for i, x in enumerate(x_list[1:]):
            loss += self.weight[i + 1] * nn.BCEWithLogitsLoss(weight=None, size_average=False)(x, label)
        loss /= n 
        loss /= len(x_list)
        return loss

if __name__ == '__main__':
    net = DSSNet()
    x = torch.randn(1, 3, 512, 512)
    y = net(x)
    print(net)
    print(y.size())