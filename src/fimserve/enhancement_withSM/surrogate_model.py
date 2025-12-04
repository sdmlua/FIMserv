#Import the libraries and packages
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pathlib import Path
from PIL import Image
import pandas as pd
import torchmetrics
from torchmetrics import Metric
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
import torch.nn.functional as F

#IOU loss
class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        return 1 - iou
    
# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Attention U-Net Model Initialization
class AttentionUNet(pl.LightningModule):
    def __init__(self, channel: int, num_classes: int = 2, task: str = 'binary', lr: float = 0.001, weight_decay : float = 0.0001) -> None:
        super().__init__()

        # Defining the model
        self.channel = channel
        self.lr = lr
        self.task = task
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        # Encoder
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = self._conv_block(self.channel, 64)
        self.conv2 = self._conv_block(64, 128)
        self.conv3 = self._conv_block(128, 256)
        self.conv4 = self._conv_block(256, 512)
        self.conv5 = self._conv_block(512, 1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attn1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.upconv_block1 = self._conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.upconv_block2 = self._conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.upconv_block3 = self._conv_block(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.upconv_block4 = self._conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        # Define metric
        alpha = 0.8
        beta = 0.2
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        # self.loss_fn = lambda x, y: alpha * self.bce(x, y) + beta * IoULoss()(x, y)
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=self.num_classes)

        # Initialize containers for predictions and labels
        self.val_preds = []
        self.val_labels = []

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        p1 = self.max_pool(e1)
        e2 = self.conv2(p1)
        p2 = self.max_pool(e2)
        e3 = self.conv3(p2)
        p3 = self.max_pool(e3)
        e4 = self.conv4(p3)
        p4 = self.max_pool(e4)
        e5 = self.conv5(p4)

        # Decoder with Attention
        d1 = self.upconv1(e5)
        attn1 = self.attn1(g=d1, x=e4)
        d1 = torch.cat((attn1, d1), dim=1)
        d1 = self.upconv_block1(d1)

        d2 = self.upconv2(d1)
        attn2 = self.attn2(g=d2, x=e3)
        d2 = torch.cat((attn2, d2), dim=1)
        d2 = self.upconv_block2(d2)

        d3 = self.upconv3(d2)
        attn3 = self.attn3(g=d3, x=e2)
        d3 = torch.cat((attn3, d3), dim=1)
        d3 = self.upconv_block3(d3)

        d4 = self.upconv4(d3)
        attn4 = self.attn4(g=d4, x=e1)
        d4 = torch.cat((attn4, d4), dim=1)
        d4 = self.upconv_block4(d4)

        output = self.out_conv(d4)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        acc = self.accuracy(y_hat, y)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        acc = self.accuracy(y_hat, y)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    #PLATEAU LEARNING RATE
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',               
            factor=0.5,              
            patience=3,              
            min_lr=1e-6,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'interval': 'epoch',
                'frequency': 1
            }
        }
