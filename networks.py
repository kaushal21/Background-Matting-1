import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class Network(nn.Module):
    def __init__(self, input, output, ngf=64, nf_part=64, padding='reflect', normalization=nn.BatchNorm2d, dropout=False, n_block1=5, n_block2=2):
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
                              normalization(int(ngf * nextICMulti / 2)),
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
        img1 = self.model_encoder1(image)
        img = self.model_encoder2(img1)

        back = self.model_back_encoder(background)
        seg = self.model_seg_encoder(segmentation)

        out_cat = torch.cat([self.comb_back(torch.cat([img, back], dim=1)),
                             self.comb_seg(torch.cat([img, seg], dim=1)),
                             self.comb_multi(torch.cat([img, back], dim=1))], dim=1)

        out = self.model_resnet_decoder(torch.cat([img, out_cat], dim=1))

        alpha_dec = self.model_resnet_alpha_decoder(out)
        alpha_out = self.model_alpha_output(alpha_dec)

        foreground_dec = self.model_resnet_foreground_decoder(out)
        foreground_dec1 = self.model_foreground1_decoder(foreground_dec)
        foreground_out = self.model_foreground_output(torch.cat([foreground_dec1, img1], dim=1))

        return alpha_out, foreground_out

############################## Resnet Block ##############################


class ResBLK(nn.Module):
    def __init__(self, dimension, padding, normalization, dropout, bias):
        super(ResBLK, self).__init__()
        self.convolutionBlock = self.build_convolutionBlock(dimension=dimension, padding=padding, normalization=normalization, dropout=dropout, bias=bias)

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
        init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        if model.bias is not None:
            init.constant_(model.bias, 0)

    if classname.find('Linear') != -1:
        init.normal_(model.weight)
        init.constant_(model.bias, 1)

    if classname.find('BatchNorm2d') != -1:
        init.normal_(model.weight.data, 1.0, 0.2)
        init.constant_(model.bias, 0.0)


############################## Discriminator ##############################


class WGANDiscriminator(nn.Module):
    def __init__(self, input, ndf=46, layers=3, normalization=nn.BatchNorm2d, sigmoid=False, numD=3, getIntermediateFeat=False):
        super(WGANDiscriminator).__init__()
        pass

        self.numD = numD
        self.layers = layers
        self.getIntermediateFeat = getIntermediateFeat

        for i in range(self.numD):
            netD = DiscriminatorLayer(input, ndf, layers, normalization, sigmoid, getIntermediateFeat)
            if getIntermediateFeat:
                for j in range(layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downSample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def fwd1(self, model, input):
        if self.getIntermediateFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self):
        num_D = self.num_D
        result = []
        input_downSampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.fwd1(model, input_downSampled))
            if i != (num_D - 1):
                input_downsampled = self.downSample(input_downSampled)
        return result


class DiscriminatorLayer(nn.Module):
    def __init__(self, input, ndf=64, layers=3, normalization=nn.BatchNorm2d, sigmoid=False, getIntermediateFeat=False):
        self.getIntermFeat = getIntermediateFeat
        self.n_layers = layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[
            nn.Conv2d(input, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                normalization(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            normalization(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermediateFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)



