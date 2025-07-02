import torch
from torch import nn
import torch.nn.functional as F

class TriggerDisentangle(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(TriggerDisentangle, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        self.victim_layer = nn.Linear(input_dim, self.C * self.H * self.W, bias=False).to(device)
        self.target_layer = nn.Linear(input_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        victim_mask = self.victim_layer(feature[:, :feature.size(1) // 2]).view(-1, self.C, self.H, self.W)
        victim_mask = F.sigmoid(victim_mask) * self.epsilon / 2
        target_mask = self.target_layer(feature[:, feature.size(1) // 2:]).view(-1, self.C, self.H, self.W)
        target_mask = F.sigmoid(target_mask) * self.epsilon / 2

        return victim_mask + target_mask

