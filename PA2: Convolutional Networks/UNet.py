import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class FCN_Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # Encoder
        self.enc_block1 = Block(3, 32)
        self.enc_block2 = Block(32, 64)
        self.enc_block3 = Block(64, 128)
        self.enc_block4 = Block(128, 256)
        self.enc_block5 = Block(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_block1 = Block(512, 256)  

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_block2 = Block(256, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_block3 = Block(128, 64)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_block4 = Block(64, 32)

        # Upsampling Refinement (Bilinear + Extra Conv)
        self.refine1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.refine3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.refine4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Dropout
        self.dropout = nn.Dropout2d(0.2)

        # Final classifier
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_block1(x)  
        enc2 = self.enc_block2(self.maxpool(enc1))  
        enc3 = self.enc_block3(self.maxpool(enc2))  
        enc4 = self.enc_block4(self.maxpool(enc3))  
        enc5 = self.enc_block5(self.maxpool(enc4))  

        # Decoder with cropping for alignment
        dec1 = self.upconv1(enc5)  
        enc4_cropped = self.crop(enc4, dec1.shape[2], dec1.shape[3])
        dec1 = torch.cat([dec1, enc4_cropped], dim=1)  
        dec1 = self.dropout(self.dec_block1(dec1))  
        dec1 = self.refine1(dec1)  # Extra conv layer for refinement

        dec2 = self.upconv2(dec1)  
        enc3_cropped = self.crop(enc3, dec2.shape[2], dec2.shape[3])
        dec2 = torch.cat([dec2, enc3_cropped], dim=1)
        dec2 = self.dropout(self.dec_block2(dec2))  
        dec2 = self.refine2(dec2)

        dec3 = self.upconv3(dec2)  
        enc2_cropped = self.crop(enc2, dec3.shape[2], dec3.shape[3])
        dec3 = torch.cat([dec3, enc2_cropped], dim=1)
        dec3 = self.dropout(self.dec_block3(dec3))  
        dec3 = self.refine3(dec3)

        dec4 = self.upconv4(dec3)  
        enc1_cropped = self.crop(enc1, dec4.shape[2], dec4.shape[3])
        dec4 = torch.cat([dec4, enc1_cropped], dim=1)
        dec4 = self.dropout(self.dec_block4(dec4))  
        dec4 = self.refine4(dec4)

        # Final classifier
        score = self.classifier(dec4)  

        return score

    def crop(self, tnsr, target_h, target_w):
        """ Center crop an input tensor to match target height and width """
        _, _, h, w = tnsr.shape
        delta_h = (h - target_h) // 2
        delta_w = (w - target_w) // 2
        return tnsr[:, :, delta_h:delta_h + target_h, delta_w:delta_w + target_w]
