# # LeakyRelu
# import torch.nn as nn

# # ToDO Fill in the __ values
# class FCN(nn.Module):

#     def __init__(self, n_class):
#         super().__init__()
#         self.n_class = n_class
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd4 = nn.BatchNorm2d(256)
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
#         self.bnd5 = nn.BatchNorm2d(512)

#         # Changed from ReLU to LeakyReLU
#         self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn1 = nn.BatchNorm2d(512)
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn5 = nn.BatchNorm2d(32)
#         self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

#     def forward(self, x):
#         x1 = self.bnd1(self.activation(self.conv1(x)))
#         x2 = self.bnd2(self.activation(self.conv2(x1)))
#         x3 = self.bnd3(self.activation(self.conv3(x2)))
#         x4 = self.bnd4(self.activation(self.conv4(x3)))
#         x5 = self.bnd5(self.activation(self.conv5(x4)))

#         y1 = self.bn1(self.activation(self.deconv1(x5)))
#         y2 = self.bn2(self.activation(self.deconv2(y1)))
#         y3 = self.bn3(self.activation(self.deconv3(y2)))
#         y4 = self.bn4(self.activation(self.deconv4(y3)))
#         y5 = self.bn5(self.activation(self.deconv5(y4)))

#         score = self.classifier(y5)

#         return score  # size=(N, n_class, H, W)

# #--------------------
# changing filter sizes
# import torch.nn as nn
# import torch

# class FCN(nn.Module):
#     def __init__(self, n_class):
#         super().__init__()
#         self.n_class = n_class

#         # Encoder
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)  # 5x5
#         self.bnd1 = nn.BatchNorm2d(64)
        
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # 5x5
#         self.bnd2 = nn.BatchNorm2d(128)
        
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 3x3
#         self.bnd3 = nn.BatchNorm2d(256)
        
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 3x3
#         self.bnd4 = nn.BatchNorm2d(512)
        
#         self.conv5 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)  # 1x1
#         self.bnd5 = nn.BatchNorm2d(1024)

#         # Decoder 
#         self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)  # 3x3
#         self.bn1 = nn.BatchNorm2d(512)
        
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # 3x3
#         self.bn2 = nn.BatchNorm2d(256)
        
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)  # 5x5
#         self.bn3 = nn.BatchNorm2d(128)
        
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)  # 5x5
#         self.bn4 = nn.BatchNorm2d(64)
        
#         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1)  # 7x7
#         self.bn5 = nn.BatchNorm2d(32)

#         # 1x1 conv 
#         self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
#         self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

#     def forward(self, x):
#         # Encoder
#         x1 = self.bnd1(self.activation(self.conv1(x)))
#         x2 = self.bnd2(self.activation(self.conv2(x1)))
#         x3 = self.bnd3(self.activation(self.conv3(x2)))
#         x4 = self.bnd4(self.activation(self.conv4(x3)))
#         x5 = self.bnd5(self.activation(self.conv5(x4)))

#         # Decoder
#         y1 = self.bn1(self.activation(self.deconv1(x5)))
#         y2 = self.bn2(self.activation(self.deconv2(y1)))
#         y3 = self.bn3(self.activation(self.deconv3(y2)))
#         y4 = self.bn4(self.activation(self.deconv4(y3)))
#         y5 = self.bn5(self.activation(self.deconv5(y4)))

#         score = self.classifier(y5)  # Output layer

#         score = torch.nn.functional.interpolate(score, size=(224, 224), mode='bilinear', align_corners=False)

#         return score  # (N, n_class, 224, 224)


#---------------
# changing layer size
import torch.nn as nn
import torch

class FCN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 7x7
        self.bnd1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # 5x5
        self.bnd2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 3x3
        self.bnd3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 3x3
        self.bnd4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)  # 1x1
        self.bnd5 = nn.BatchNorm2d(1024)

        self.conv6 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bnd6 = nn.BatchNorm2d(1024)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.deconv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x1 = self.bnd1(self.activation(self.conv1(x)))
        x2 = self.bnd2(self.activation(self.conv2(x1)))
        x3 = self.bnd3(self.activation(self.conv3(x2)))
        x4 = self.bnd4(self.activation(self.conv4(x3)))
        x5 = self.bnd5(self.activation(self.conv5(x4)))
        x6 = self.bnd6(self.activation(self.conv6(x5)))

        y1 = self.bn1(self.activation(self.deconv1(x6)))
        y2 = self.bn2(self.activation(self.deconv2(y1)))
        y3 = self.bn3(self.activation(self.deconv3(y2)))
        y4 = self.bn4(self.activation(self.deconv4(y3)))
        y5 = self.bn5(self.activation(self.deconv5(y4)))
        y6 = self.bn6(self.activation(self.deconv6(y5)))

        score = self.classifier(y6)  

        score = torch.nn.functional.interpolate(score, size=(224, 224), mode='bilinear', align_corners=False)

        return score  # (N, n_class, 224, 224)





