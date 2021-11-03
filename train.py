import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_loader import AdobeData
from loss_function import AlphaLoss, ComposeLoss, AlphaGradientLoss
from networks import Network, conv_init


# python train.py -n train-adobe-40 -bs 4 -res 256x256 -epoch 40 -n_blocks1 5 -n_blocks 2
parser = argparse.ArgumentParser(description='Training Background Matting on Adobe Dataset.')
parser.add_argument('-n', '--name', type=str, help='Name of tensorboard and model saving folders.')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch Size.')
parser.add_argument('-res', '--resolution', type=int, help='Input image resolution')

parser.add_argument('-epoch', '--epoch', type=int, default=60, help='Maximum Epoch')
parser.add_argument('-n_blocks1', '--n_blocks1', type=int, default=5, help='Number of residual blocks after Context Switching.')
parser.add_argument('-n_blocks2', '--n_blocks2', type=int, default=2, help='Number of residual blocks for Fg and alpha each.')

args = parser.parse_args()

# Model Data Path
model_dir = 'Models/' + args.name

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

############################ Loading Data ############################
print('\n[Phase 1] : Data Preparation')

def collate_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# Data Config for Data Loader
config = {'trimapK': [5, 5], 'resolution': [args.resolution, args.resolution], 'noise': True}

# Load Original Data from csv file using data loader
traindata = AdobeData(file='Data_adobe/Adobe_train_data.csv', config=config)

train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.batch_size, collate_fn=collate_filter_none)

############################ Initializing the Network ############################
print('\n[Phase 2] : Initialization')

network = Network(input=(3, 3, 1, 4), output=4, n_block1=5, n_block2=2, normalization=nn.BatchNorm2d)
network.apply(conv_init)
network = nn.DataParallel(network)
# net.load_state_dict(torch.load(model_dir + 'net_epoch_X')) #uncomment this if you are initializing your model
network.cuda()
torch.backends.cudnn.benchmark = True

############################ Initializing the Losses and Optimizer ############################
alpha_loss = AlphaLoss()
comp_loss = ComposeLoss()
alpha_grad_loss = AlphaGradientLoss()

optimizer = optim.Adam(network.parameters(), lr=1e-4)
# optimizer.load_state_dict(torch.load(model_dir + 'optim_epoch_X')) #uncomment this if you are initializing your model

############################ Start Training ############################
print('\n[Phase 3] : Starting Training')
step = 50  # steps to visualize training images in tensorboard

KK = len(train_loader)

for epoch in range(0, args.epoch):
    network.train()         # Make the network in training mode

    # Initialize the losses
    networkLoss, alphaLoss, foregroundLoss, foregroundComposeLoss, alphaGradLoss, networkRunTime, iterationRunTime = 0, 0, 0, 0, 0, 0, 0

    startTime = time.time()
    testL = 0           # Loss for each epoch
    ct_tst = 0          # Steps done in the dataloader
    for i, data in enumerate(train_loader):
        # Get all the images from data
        fg = data['fg']; fg = Variable(fg.cuda())
        bg = data['bg']; bg = Variable(bg.cuda())
        alpha = data['alpha']; alpha = Variable(alpha.cuda())
        image = data['image']; image = Variable(image.cuda())
        segmentation = data['segmentation']; segmentation = Variable(segmentation.cuda())
        bg_tr = data['bg_tr']; bg_tr = Variable(bg_tr.cuda())

        mask = (alpha > -0.99).type(torch.cuda.FloatTensor)
        mask0 = Variable(torch.ones(alpha.shape).cuda())

        networkStartTime = time.time()

        alpha_pred, fg_pred = network(image, bg_tr, segmentation)

        # Calculate the Losses
        al_loss = alpha_loss(alpha, alpha_pred, mask0)
        fg_loss = alpha_loss(fg, fg_pred, mask)

        # Get the alpha mask from the alpha prediction. Keep only the values greater then threshold
        # Calculate the foreground using alpha mask
        al_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)
        fg_calc = image * al_mask + fg_pred * (1 - al_mask)

        # Compute Compose Loss and Alpha Gradient Loss
        fg_calc_comp_loss = comp_loss(image, alpha_pred, fg_calc, bg, mask0)
        al_grad_loss = alpha_grad_loss(alpha, alpha_pred, mask0)

        # Total Loss
        loss = al_loss + 2 * fg_loss + fg_calc_comp_loss + al_grad_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the total loss for this epoch
        networkLoss += loss.data
        alphaLoss += al_loss.data
        foregroundLoss += fg_loss.data
        foregroundComposeLoss += fg_calc_comp_loss.data
        alphaGradLoss += al_grad_loss.data
        testL += loss.data
        ct_tst += 1

        # Time the run of each iteration
        endTime = time.time()
        iterationRunTime += endTime - startTime
        networkRunTime += endTime - networkStartTime
        startTime = endTime

        # Print the Losses at an interval of step
        if i % step == (step - 1):
            print('[', epoch + 1, ', ', i + 1, ']',
                  ' Total-loss:  ', networkLoss / step,
                  ' Alpha-loss: ', alphaLoss / step,
                  ' Fg-loss: ', foregroundLoss / step,
                  ' Comp-loss: ', foregroundComposeLoss / step,
                  ' Alpha-gradient-loss: ', alphaGradLoss / step,
                  ' Time-all: ', iterationRunTime / step,
                  ' Time-fwbw: ', networkRunTime / step)

            networkLoss, alphaLoss, foregroundLoss, foregroundComposeLoss, alphaGradLoss, networkRunTime, iterationRunTime = 0, 0, 0, 0, 0, 0, 0

            # Calculating the Composing image from foreground, alpha matte and background
            alpha_pred = (alpha_pred + 1) / 2
            comp = fg_pred * alpha_pred + (1 - alpha_pred) * bg

            del comp

        del fg, bg, alpha, image, alpha_pred, fg_pred, segmentation

    # Saving
    torch.save(network.state_dict(), model_dir + '/net_epoch_%d.pth' % epoch)
    torch.save(optimizer.state_dict(), model_dir + '/optim_epoch_%d.pth' % epoch)
