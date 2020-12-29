import numpy as np
import os
import sys
import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = "to_be_given"
dir_mask = "to_be_given"
dir_checkpoint = "to_be_given"

def train_net(net, device, epochs = 10, batch_size = 16, lr = 0.0001, val_percent = 0.1, save_cp = True, img_scale = 0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(dataset = train, shuffle = False, batch_size = batch_size, num_workers = 8, pin_memory = True)
    val_loader = DataLoader(dataset = val, shuffle = False, batch_size = batch_size, pin_memory = True, drop_last = True, num_workers = 8)

    writer = SummaryWriter(comment = f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr = lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, mode = 'min' if net.n_classes > 1 else 'max')

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total = n_train, desc = f'Epoch {epoch+1} / {epochs}', unit = 'img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                imgs = imgs.to(device, dtype = torch.float32)
                mask_type = torch.float32 if net.classes == 1 else torch.long
                true_masks = mask_type.to(device, dtype = mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)':loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lt'], global_step)

                    if net.n_classes > 1:
                        logging.info('Valudation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Valudation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.classes == 1:
                        writer.add_images('mask/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, format = '%(levelname)s: (message)s')
    device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels = 3, n_classes = 1, bilinear = True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.to(device = device)

    try:
        train_net(net = net, epochs = 10, batch_size = 16, lr = 0.0001, img_scale=0.5, device = device, val_percent=0.1)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
