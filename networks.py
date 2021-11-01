import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class Network(nn.Module):
    def __init__(self, input, output, ngf=64, nf_part=64, padding='reflect', normalization=nn.BatchNorm2d, dropout=False, n_block1=7, n_block2=3):
        # Check for block sizes
        assert n_block1 >= 0
        assert n_block2 >= 0

        super(Network, self).__init__()
        self.input_layer = input
        self.output_layer = output
        self.ngf = ngf
        bias = True

        # Main Encoder
        encoder1 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input[0], ngf, kernel_size=(7, 7), padding=0, bias=bias),
                    normalization(ngf),
                    nn.ReLU(True),
                    nn.Conv2d(ngf, ngf * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=bias),
                    normalization(ngf * 2),
                    nn.ReLU(True)]
        encoder2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=bias),
                    normalization(ngf * 4),
                    nn.ReLU(True)]

        # Back Encoder
        back_encoder = [nn.ReflectionPad2d(3),
                        nn.Conv2d(input[1], ngf, kernel_size=(7, 7), padding=0, bias=bias),
                        normalization(ngf),
                        nn.ReLU(True)]
        for i in range(2):
            nextICMulti = 2 ** i
            back_encoder += [nn.Conv2d(ngf * nextICMulti, ngf * nextICMulti * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=bias),
                             normalization(ngf * nextICMulti * 2),
                             nn.ReLU(True)]

        # Segmentation Encoder
        seg_encoder = [nn.ReflectionPad2d(3),
                       nn.Conv2d(input[2], ngf, kernel_size=(7, 7), padding=0, bias=bias),
                       normalization(ngf),
                       nn.ReLU(True)]
        for i in range(2):
            nextICMulti = 2 ** i
            seg_encoder += [nn.Conv2d(ngf * nextICMulti, ngf * nextICMulti * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=bias),
                            normalization(ngf * nextICMulti * 2),
                            nn.ReLU(True)]

        # ResNet Decoder
        nextICMulti = 2 ** 2
        resnet_decoder = [nn.Conv2d(ngf * nextICMulti + 3 * nf_part, ngf * nextICMulti, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                         normalization(ngf * nextICMulti),
                         nn.ReLU(True)]
        for i in range(n_block1):
            resnet_decoder += [ResBLK(
                dimension=ngf * nextICMulti,
                padding=padding,
                normalization=normalization,
                dropout=dropout,
                bias=bias)]

        # ResNet Alpha Decoder
        resnet_alpha_decoder = []
        for i in range(n_block2):
            resnet_alpha_decoder += [ResBLK(
                dimension=ngf * nextICMulti,
                padding=padding,
                normalization=normalization,
                dropout=dropout,
                bias=bias)]

        # ResNet Foreground Decoder
        resnet_foreground_decoder = []
        for i in range(n_block2):
            resnet_foreground_decoder += [ResBLK(
                dimension=ngf * nextICMulti,
                padding=padding,
                normalization=normalization,
                dropout=dropout,
                bias=bias)]

        # Alpha Decoder
        alpha_decoder = []
        for i in range(2):
            nextICMulti = 2 ** (2 - i)
            alpha_decoder += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                              nn.Conv2d(ngf * nextICMulti, int(ngf * nextICMulti / 2), kernel_size=(3, 3), stride=(1, 1), padding=1),
                              normalization(int(ngf * nextICMulti / 3)),
                              nn.ReLU(True)]
        alpha_decoder += [nn.ReflectionPad2d(3),
                          nn.Conv2d(ngf, 1, kernel_size=(7, 7), padding=0),
                          nn.Tanh()]

        # Foreground Decoder
        foreground_decoder1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                               nn.Conv2d(ngf * 4, ngf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
                               normalization(ngf * 2),
                               nn.ReLU(True)]
        foreground_decoder2 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                               nn.Conv2d(ngf * 4, ngf, kernel_size=(3, 3), stride=(1, 1), padding=1),
                               normalization(ngf),
                               nn.ReLU(True),
                               nn.ReflectionPad2d(3),
                               nn.Conv2d(ngf, output-1, kernel_size=(7, 7), padding=0)]

        # Set Model Parameters
        self.model_encoder1 = nn.Sequential(*encoder1)
        self.model_encoder2 = nn.Sequential(*encoder2)
        self.model_back_encoder = nn.Sequential(*back_encoder)
        self.model_seg_encoder = nn.Sequential(*seg_encoder)

        nextICMulti = 2 ** 2
        self.comb_back = nn.Sequential(nn.Conv2d(ngf * nextICMulti * 2, nf_part, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                                       normalization(ngf),
                                       nn.ReLU(True))
        self.comb_seg = nn.Sequential(nn.Conv2d(ngf * nextICMulti * 2, nf_part, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                                      normalization(ngf),
                                      nn.ReLU(True))
        self.comb_multi = nn.Sequential(nn.Conv2d(ngf * nextICMulti * 2, nf_part, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                                        normalization(ngf),
                                        nn.ReLU(True))

        self.model_resnet_decoder = nn.Sequential(*resnet_decoder)

        self.model_resnet_alpha_decoder = nn.Sequential(*resnet_alpha_decoder)
        self.model_alpha_output = nn.Sequential(*alpha_decoder)

        self.model_resnet_foreground_decoder = nn.Sequential(*resnet_foreground_decoder)
        self.model_foreground1_decoder = nn.Sequential(*foreground_decoder1)
        self.model_foreground_output = nn.Sequential(*foreground_decoder2)

    def forward(self, image, background, segmentation):
        img = self.model_encoder1(image)
        img = self.model_encoder2(img)

        back = self.model_back_encoder(background)
        seg = self.model_seg_encoder(segmentation)

        out = torch.cat([self.comb_back(torch.cat([img, back], dim=1)),
                         self.comb_seg(torch.cat([img, seg], dim=1)),
                         self.comb_multi(torch.cat([img, back], dim=1))], dim=1)

        out = self.model_resnet_decoder(torch.cat([img, out], dim=1))

        alpha_dec = self.model_resnet_alpha_decoder(out)
        alpha_out = self.model_alpha_output(alpha_dec)

        foreground_dec = self.model_resnet_foreground_decoder(out)
        foreground_dec1 = self.model_foreground1_decoder(foreground_dec)
        foreground_out = self.model_foreground_output(foreground_dec1)

        return alpha_out, foreground_out

############################## Resnet Block ##############################


class ResBLK(nn.Module):
    def __init__(self, dimension, padding, normalization, dropout, bias):
        super(ResBLK).__init__()
        self.convolutionBlock = self.build_convolutionBlock(dimension, padding, normalization, dropout, bias)
        pass

    def build_convolutionBlock(self, dimension, padding, normalization, dropout, bias):
        block = []
        # Adding Padding
        pad = 0
        if padding == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            block += [nn.ReflectionPad2d(1)]
        elif padding == 'zero':
            pad = 0
        else:
            raise NotImplementedError('Padding is not Implemented: ', padding)

        # Adding Convolution Block, Normalization and ReLU
        block += [nn.Conv2d(dimension, dimension, kernel_size=(3, 3), padding=pad, bias=bias), normalization(dimension), nn.ReLU(True)]

        # Adding Dropout
        if dropout:
            block += [nn.Dropout(0.5)]

        # Adding Padding
        pad = 0
        if padding == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            block += [nn.ReflectionPad2d(1)]
        elif padding == 'zero':
            pad = 0
        else:
            raise NotImplementedError('Padding is not Implemented: ', padding)

        # Adding Convolution Block and Normalization
        block += [nn.Conv2d(dimension, dimension, kernel_size=(3, 3), padding=pad, bias=bias), normalization(dimension)]

        return nn.Sequential(*block)

    def forward(self, x):
        output = x + self.convolutionBlock(x)
        return output

############################## Convolution Initialization ##############################


def conv_init(model):
    """
    Initialize the model based on the class it is from.
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(model.weight, gain=np.sqrt(2))
        if model.bias is not None:
            init.constant(model.bias, 0)

    if classname.find('Linear') != -1:
        init.normal(model.weight)
        init.constant(model.bias, 1)

    if classname.find('BatchNorm2d') != -1:
        init.normal(model.weight.data, 1.0, 0.2)
        init.constant(model.bias, 0.0)


############################## Discriminator ##############################


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator).__init__()
        pass

    def forward(self):
        pass


class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super(NLayerDiscriminator).__init__()
        pass

    def forward(self):
        pass



