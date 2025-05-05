import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16]  # Up to relu_2_2
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # VGG expects these
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, pred, target):
        # Assume input is [-1, 1] → convert to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        # Resize if needed
        if self.resize:
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize for VGG
        pred = self.normalize(pred)
        target = self.normalize(target)

        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)

        return F.l1_loss(pred_feat, target_feat)
