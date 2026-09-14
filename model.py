import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
import modules
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models as light_models
from torchvision.models import MobileNet_V2_Weights

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gazefeature = modules.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.facefeature = modules.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.channel = 1280  # 同步修改为 MobileNetV2 输出通道数

        self.featuredim = 2048
        self.fusion = nn.Conv2d(2*self.channel, self.featuredim, 1)
        # self.feature.load_state_dict(torch.load(pretrained_url), strict=False )

        self.gazeEs = modules.ResGazeEs(channel=self.channel)
        self.featureloss = featureloss(w1=0.3, w2=0.3, w3=0.3)
        # self.gazeEs.load_state_dict(torch.load(pretrained_url), strict=False )

        self.deconv = modules.ResDeconv(modules.BasicBlock)

    def forward(self, x_in, require_img=True):
        gazefeatures = self.gazefeature(x_in["face"])
        facefeatures = self.facefeature(x_in["face"])
        
        gazepairedfeatures = self.gazefeature(x_in["pairedface"])
        facepairedfeatures = self.facefeature(x_in["pairedface"])
        
        featureloss = self.featureloss(gazefeatures, facefeatures, gazepairedfeatures, facepairedfeatures)
        
        deltagazefeature = gazefeatures - gazepairedfeatures
        
        face1 = self.fusion(torch.cat((gazefeatures, facepairedfeatures), dim=1))
        face2 = self.fusion(torch.cat((gazepairedfeatures, facefeatures), dim=1))
        
        gaze = self.gazeEs(gazefeatures)
        pairedgaze = self.gazeEs(gazepairedfeatures)
        deltagaze = self.gazeEs(deltagazefeature)
        
        if require_img:
          img1 = self.deconv(face1)
          img1 = torch.sigmoid(img1)
          img2 = self.deconv(face2)
          img2 = torch.sigmoid(img2)
        else:
          img1 = None
          img2 = None
        return gaze, pairedgaze, deltagaze, img1, img2, featureloss






# ================= 1. 视线回归损失 (Gelossop) =================
class Gelossop(nn.Module):
    def __init__(self, w1=1.0, w2=1.0, w3=1.0, dead_zone=0.03, max_clip=0.5):
        super().__init__()
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.dead_zone = dead_zone
        self.max_clip = max_clip

    def forward(self, gaze, pairedgaze, deltagaze, label, pairedlabel):
        loss1 = torch.norm(gaze - label, p=2, dim=-1).mean()
        loss2 = torch.norm(pairedgaze - pairedlabel, p=2, dim=-1).mean()

        label_diff = label - pairedlabel
        diff_norm = torch.norm(label_diff, p=2, dim=-1, keepdim=True)
        valid_mask = (diff_norm > self.dead_zone).float()
        clipped_diff = torch.clamp(label_diff, -self.max_clip, self.max_clip)

        delta_error = torch.norm(deltagaze - clipped_diff, p=2, dim=-1, keepdim=True)
        masked_loss = (delta_error * valid_mask).sum()
        denom = valid_mask.sum().clamp(min=1.0)
        loss3 = masked_loss / denom

        return self.w1 * loss1 + self.w2 * loss2 + self.w3 * loss3


# ================= 2. 图像重建损失 (Delossop) =================
class Delossop(nn.Module):
    def __init__(self, w_ssim=0.8, w_l1=0.2, window_size=7):
        super().__init__()
        self.w_ssim = w_ssim
        self.w_l1 = w_l1
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.window_size = window_size

    def _ssim(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 ** 2, self.window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, self.window_size, stride=1, padding=pad) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=pad) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, img1, img2, img, pairedimg):
        l1 = self.l1_loss(img1, img) + self.l1_loss(img2, pairedimg)
        ssim = self._ssim(img1, img) + self._ssim(img2, pairedimg)
        return self.w_l1 * l1 + self.w_ssim * (2.0 - ssim)


# ================= 3. 特征约束损失 (FeatureLoss) =================
class featureloss(nn.Module):
    def __init__(self, w1=0.3, w2=0.3, w3=0.3, temperature=5.0):
        super().__init__()
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.temperature = temperature
        self.ada = nn.AdaptiveAvgPool2d(1)

    def forward(self, gaze1, face1, gaze2, face2):
        g1 = self.ada(gaze1).flatten(1)
        f1 = self.ada(face1).flatten(1)
        g2 = self.ada(gaze2).flatten(1)
        f2 = self.ada(face2).flatten(1)

        face_sim = F.cosine_similarity(f1, f2, dim=-1)
        face_loss = (1 - face_sim).mean()

        gf1_sim = F.cosine_similarity(g1, f1, dim=-1) * self.temperature
        gf2_sim = F.cosine_similarity(g2, f2, dim=-1) * self.temperature
        gf_loss = (gf1_sim.abs().mean() + gf2_sim.abs().mean()) / 2.0

        return self.w1 * face_loss + (self.w2 + self.w3) * gf_loss

def test_featureloss():
    """测试特征约束损失"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face1 = torch.randn(10, 1024, 7, 7, device=device)
    gaze1 = torch.randn(10, 1024, 7, 7, device=device)
    face2 = torch.randn(10, 1024, 7, 7, device=device)
    gaze2 = torch.randn(10, 1024, 7, 7, device=device)

    floss = featureloss(w1=0.3, w2=0.3, w3=0.3).to(device)
    loss = floss(gaze1, face1, gaze2, face2)
    print(f"[FeatureLoss] loss = {loss.item():.6f}")


def test_model_forward():
    """测试模型前向传播"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Model().to(device)
    test_input = {
        "face": torch.randn(2, 3, 224, 224, device=device),
        "pairedface": torch.randn(2, 3, 224, 224, device=device),
    }
    gaze, pairedgaze, deltagaze, img1, img2, floss = net(test_input)
    print(f"[Model Forward] gaze shape: {gaze.shape}")
    print(f"[Model Forward] feature channel: {net.channel}")


if __name__ == "__main__":
    # 👇 按需注释/取消注释，灵活选择要运行的测试
    test_featureloss()
    # test_model_forward()
