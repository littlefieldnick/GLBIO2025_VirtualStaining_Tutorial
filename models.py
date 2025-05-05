import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_channels=3):
        super(Generator, self).__init__()

        # Encoder Layers: 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 -> 512
        self.enc1 = self.conv_block(input_channels, 64)    
        self.enc2 = self.conv_block(64, 128)  
        self.enc3 = self.conv_block(128, 256) 
        self.enc4 = self.conv_block(256, 512) 
        self.enc5 = self.conv_block(512, 512) 
        self.enc6 = self.conv_block(512, 512) 
        self.enc7 = self.conv_block(512, 512) 

        # U-Net Decoder layers 512 -> 1024 -> 1024 -> 1024 -> 1024 -> 512 -> 256 -> 128
        self.up1 = self.deconv_block(512, 512)   
        self.up2 = self.deconv_block(1024, 1024)  # Fix the input channels here (1024 + 512)
        self.up3 = self.deconv_block(1024 + 512, 1024)  # Fix the input channels here (1024 + 256)
        self.up4 = self.deconv_block(1024 + 512, 1024)  # Fix the input channels here (1024 + 128)
        self.up5 = self.deconv_block(1024 + 256, 512)    # Fix the input channels here (1024 + 64)
        self.up6 = self.deconv_block(512 + 128, 256)     # Fix the input channels here (512 + 64)
        self.up7 = self.deconv_block(256 + 64, 128)     # Fix the input channels here (256 + 64)

        # Final output layer (RGB output)
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)
        enc6_out = self.enc6(enc5_out)
        enc7_out = self.enc7(enc6_out)

        # Decoder with skip connections
        up1_out = self.up1(enc7_out)  # This will be 512 channels
        up2_out = self.up2(torch.cat([up1_out, enc6_out], 1))  # Concatenate skip connection (1024 + 512 = 1536 channels)
        up3_out = self.up3(torch.cat([up2_out, enc5_out], 1))  # Concatenate skip connection (1024 + 256 = 1280 channels)
        up4_out = self.up4(torch.cat([up3_out, enc4_out], 1))  # Concatenate skip connection (1024 + 128 = 1152 channels)
        up5_out = self.up5(torch.cat([up4_out, enc3_out], 1))  # Concatenate skip connection (1024 + 64 = 1088 channels)
        up6_out = self.up6(torch.cat([up5_out, enc2_out], 1))  # Concatenate skip connection (512 + 64 = 576 channels)
        up7_out = self.up7(torch.cat([up6_out, enc1_out], 1))  # Concatenate skip connection (256 + 64 = 320 channels)

        # Final output layer
        output = self.final_conv(up7_out)  # Output with 3 channels (RGB)

        return self.tanh(output)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3, dropout_p=0.3):
        super(PatchDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1 
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p), 

            # Layer 2 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p), 

            # Layer 3 
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p), 

            # Layer 4 
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),

            # Final output layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
