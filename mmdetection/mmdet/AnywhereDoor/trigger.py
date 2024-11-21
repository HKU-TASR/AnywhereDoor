import torch
from torch import nn
import torch.nn.functional as F

class Trigger1Layer(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(Trigger1Layer, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        self.linear = nn.Linear(input_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        x = self.linear(feature)
        x = x.view(-1, self.C, self.H, self.W)
        x = F.tanh(x) * self.epsilon
        return x
    
class Trigger1LayerAvgPooling(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(Trigger1LayerAvgPooling, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        # Avg pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(input_dim).to(device)

        self.linear = nn.Linear(input_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        feature = self.adaptive_pool(feature)
        x = self.linear(feature)
        x = x.view(-1, self.C, self.H, self.W)
        x = F.tanh(x) * self.epsilon
        return x
    
class Trigger1LayerMaxPooling(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(Trigger1LayerMaxPooling, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        # Max pooling
        self.adaptive_pool = nn.AdaptiveMaxPool1d(input_dim).to(device)

        self.linear = nn.Linear(input_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        feature = self.adaptive_pool(feature)
        x = self.linear(feature)
        x = x.view(-1, self.C, self.H, self.W)
        x = F.tanh(x) * self.epsilon
        return x

class Trigger3Layer(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(Trigger3Layer, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        # Define trainable layers for adapting LLM's output to image-like trigger
        self.adapter_1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.relu1 = nn.LeakyReLU()

        self.adapter_2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.relu2 = nn.LeakyReLU()

        self.adapter_3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        x = self.adapter_1(feature)
        x = self.relu1(x)

        x = self.adapter_2(x)
        x = self.relu2(x)

        x = self.adapter_3(x)

        x = x.view(-1, self.C, self.H, self.W)
        x = F.tanh(x) * self.epsilon
        return x

class Trigger3LayerAvgPooling(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(Trigger3LayerAvgPooling, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        # Avg pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(input_dim).to(device)

        # Define trainable layers for adapting LLM's output to image-like trigger
        self.adapter_1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.relu1 = nn.LeakyReLU()

        self.adapter_2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.relu2 = nn.LeakyReLU()

        self.adapter_3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        feature = self.adaptive_pool(feature)
        x = self.adapter_1(feature)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.adapter_2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.adapter_3(x)

        x = x.view(-1, self.C, self.H, self.W)
        x = F.tanh(x) * self.epsilon
        return x
    
class Trigger3LayerMaxPooling(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(Trigger3LayerMaxPooling, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        # Max pooling
        self.adaptive_pool = nn.AdaptiveMaxPool1d(input_dim).to(device)

        # Define trainable layers for adapting LLM's output to image-like trigger
        self.adapter_1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.relu1 = nn.LeakyReLU()

        self.adapter_2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.relu2 = nn.LeakyReLU()

        self.adapter_3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        feature = self.adaptive_pool(feature)
        x = self.adapter_1(feature)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.adapter_2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.adapter_3(x)

        x = x.view(-1, self.C, self.H, self.W)
        x = F.tanh(x) * self.epsilon
        return x
    
class Trigger3LayerMaxPoolingRes(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(Trigger3LayerMaxPoolingRes, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        # Max pooling
        self.adaptive_pool = nn.AdaptiveMaxPool1d(input_dim).to(device)

        # Define trainable layers for adapting LLM's output to image-like trigger
        self.adapter_1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.relu1 = nn.LeakyReLU()

        self.adapter_2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.relu2 = nn.LeakyReLU()

        self.adapter_3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

        # Define a linear layer to match the dimensions for residual connection
        self.residual_connection = nn.Linear(input_dim, hidden_dim, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        feature = self.adaptive_pool(feature)

        # Residual connection for the first layer
        residual = self.residual_connection(feature)
        x = self.adapter_1(feature)
        x = self.bn1(x)
        x = self.relu1(x)

        # Residual connection for the second layer
        x = self.adapter_2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.adapter_3(x)
        x = x + residual

        x = x.view(-1, self.C, self.H, self.W)
        x = torch.tanh(x) * self.epsilon
        return x
    
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

class TriggerDisentangle3Layer(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(TriggerDisentangle3Layer, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        self.victim_layer1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.victim_relu1 = nn.LeakyReLU()
        self.victim_layer2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.victim_relu2 = nn.LeakyReLU()
        self.victim_layer3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

        self.target_layer1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.target_relu1 = nn.LeakyReLU()
        self.target_layer2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.target_relu2 = nn.LeakyReLU()
        self.target_layer3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        victim_feature = feature[:, :feature.size(1) // 2]
        target_feature = feature[:, feature.size(1) // 2:]

        victim_feature = self.victim_layer1(victim_feature)
        victim_feature = self.victim_relu1(victim_feature)
        victim_feature = self.victim_layer2(victim_feature)
        victim_feature = self.victim_relu2(victim_feature)
        victim_feature = self.victim_layer3(victim_feature).view(-1, self.C, self.H, self.W)
        victim_mask = F.sigmoid(victim_feature) * self.epsilon / 2

        target_feature = self.target_layer1(target_feature)
        target_feature = self.target_relu1(target_feature)
        target_feature = self.target_layer2(target_feature)
        target_feature = self.target_relu2(target_feature)
        target_feature = self.target_layer3(target_feature).view(-1, self.C, self.H, self.W)
        target_mask = F.sigmoid(target_feature) * self.epsilon / 2

        return victim_mask + target_mask

class TriggerDisentangle3LayerBN(nn.Module):
    def __init__(self, epsilon, img_dim, mask_shape, input_dim, hidden_dim, device):
        super(TriggerDisentangle3LayerBN, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.C = img_dim
        self.H, self.W = mask_shape

        self.victim_layer1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.victim_bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.victim_relu1 = nn.LeakyReLU()
        self.victim_layer2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.victim_bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.victim_relu2 = nn.LeakyReLU()
        self.victim_layer3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

        self.target_layer1 = nn.Linear(input_dim, hidden_dim, bias=False).to(device)
        self.target_bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.target_relu1 = nn.LeakyReLU()
        self.target_layer2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
        self.target_bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.target_relu2 = nn.LeakyReLU()
        self.target_layer3 = nn.Linear(hidden_dim, self.C * self.H * self.W, bias=False).to(device)

    def forward(self, feature):
        feature = feature.to(self.device)
        victim_feature = feature[:, :feature.size(1) // 2]
        target_feature = feature[:, feature.size(1) // 2:]

        victim_feature = self.victim_layer1(victim_feature)
        victim_feature = self.victim_bn1(victim_feature)
        victim_feature = self.victim_relu1(victim_feature)
        victim_feature = self.victim_layer2(victim_feature)
        victim_feature = self.victim_bn2(victim_feature)
        victim_feature = self.victim_relu2(victim_feature)
        victim_feature = self.victim_layer3(victim_feature).view(-1, self.C, self.H, self.W)
        victim_mask = F.sigmoid(victim_feature) * self.epsilon / 2

        target_feature = self.target_layer1(target_feature)
        target_feature = self.target_bn1(target_feature)
        target_feature = self.target_relu1(target_feature)
        target_feature = self.target_layer2(target_feature)
        target_feature = self.target_bn2(target_feature)
        target_feature = self.target_relu2(target_feature)
        target_feature = self.target_layer3(target_feature).view(-1, self.C, self.H, self.W)
        target_mask = F.sigmoid(target_feature) * self.epsilon / 2

        return victim_mask + target_mask
