# There are total 5 Losses that we have recorded in this file
# 1. Alpha Loss
# 2. Compose Loss
# 3. Alpha Gradient Loss
# 4. Alpha Gradient Reg Loss
# 5. GAN Loss

import torch
import torch.nn as nn
import torch.nn.functional as nnFunctional
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable


def norm_l1_loss(alpha, alpha_pred, mask):
    """
    Calculating the l1 normalization loss for the given alpha
    """
    loss = 0
    e = 1e-6
    for i in range(alpha.shape[0]):
        if mask[i, ...].sum() > 0:
            loss = loss + torch.sum(torch.abs(alpha[i, ...] * mask[i, ...] - alpha_pred[i, ...] * mask[i, ...])) / (
                        torch.sum(mask[i, ...]) + e)
    loss = loss / alpha.shape[0]

    return loss


class AlphaLoss(_Loss):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, alpha, pred, mask):
        """
        Get the norm l1 loss for the given alpha
        """
        return norm_l1_loss(alpha, pred, mask)


class ComposeLoss(_Loss):
    def __init__(self):
        super(ComposeLoss, self).__init__()

    def forward(self, image, alpha, fore_ground, back_ground, mask):
        """
        Get the norm l1 loss for the image.
        For the predicted value we calculated the composed the image using F, B and pred.
        I = alpha * F + (1 - alpha) * B
        """
        alpha = (alpha + 1) / 2
        I = fore_ground * alpha + (1 - alpha) * back_ground
        return norm_l1_loss(image, I, mask)


def getGXY(alpha):
    """
    Get the 2D convolution of the alpha i.e. Gx and Gy with the weights as fx and fy.
    """
    fx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    fx = fx.view((1, 1, 3, 3))
    fx = Variable(fx.cuda())
    fy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    fy = fy.view((1, 1, 3, 3))
    fy = Variable(fy.cuda())

    G_x = nnFunctional.conv2d(alpha, fx, padding=1)
    G_y = nnFunctional.conv2d(alpha, fy, padding=1)
    return fx, fy, G_x, G_y


class AlphaGradientLoss(_Loss):
    def __init__(self):
        super(AlphaGradientLoss, self).__init__()

    def forward(self, alpha, pred, mask):
        """
        Calculate the loss as the sum of the two calculated 2D Convoluted matrix of alpha i.e., Gx and Gy
        """
        fx, fy, G_x, G_y = getGXY(alpha)
        G_x_pred = nnFunctional.conv2d(pred, fx, padding=1)
        G_y_pred = nnFunctional.conv2d(pred, fy, padding=1)
        loss = norm_l1_loss(G_x, G_x_pred, mask) + norm_l1_loss(G_y, G_y_pred, mask)

        return loss


class AlphaGradientRegLoss(_Loss):
    def __init__(self):
        super(AlphaGradientRegLoss, self).__init__()

    def forward(self, alpha, mask):
        """
        Get the average loss from the Gx and Gy convoluted matrix of alpha
        """
        fx, fy, G_x, G_y = getGXY(alpha)
        loss = (torch.sum(torch.abs(G_x)) + torch.sum(torch.abs(G_y))) / torch.sum(mask)

        return loss


class GANloss(_Loss):
    def __init__(self):
        super(GANloss, self).__init__()

    def forward(self, pred, label):
        """
        Calculate the GAN loss as the MSE loss
        """
        MSELoss = nn.MSELoss()
        loss = 0
        for i in range(0, len(pred)):
            if label:
                labels = torch.ones(pred[i][0].shape)
            else:
                labels = torch.zeros(pred[i][0].shape)
            labels = Variable(labels.cuda())

            loss += MSELoss(pred[i][0], labels)

        return loss / len(pred)
