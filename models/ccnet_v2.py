import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings

class GaborConv2d(nn.Module):
    '''
    DESCRIPTION: an implementation of the Learnable Gabor Convolution (LGC) layer \n
    INPUTS: \n
    channel_in: should be 1 \n
    channel_out: number of the output channels \n
    kernel_size, stride, padding: 2D convolution parameters \n
    init_ratio: scale factor of the initial parameters (receptive filed) \n
    '''
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding      
        self.init_ratio = init_ratio 

        if init_ratio <=0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # initial parameters
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # shape & scale of the Gaussian functioin:
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)          
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)

        # frequency of the cosine envolope:
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()    
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channelin, kernel_H, kernel_W]   
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1) 
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize) 

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # Rotated coordinate systems
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))  
                
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
    
        gb = gb - gb.mean(dim=[2,3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.theta, self.f, self.psi)
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    '''
    DESCRIPTION: an implementation of the Competitive Block
    '''
    def __init__(self, channel_in, n_competitor, ksize, stride, padding, weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp, self).__init__()

        self.channel_in = channel_in
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio

        self.gabor_conv2d = GaborConv2d(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                        padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                         padding=ksize // 2, init_ratio=init_ratio)

        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        
        self.conv1_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight
        self.weight_spa = (1-weight) / 2

    def forward(self, x):
        #1-st order
        x = self.gabor_conv2d(x)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        x_1 = self.se1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        #2-nd order
        x = self.gabor_conv2d2(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)
        x_2 = self.se2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0],-1), x_2.view(x_2.shape[0],-1)), dim=1)
        return xx


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)       
            
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
            output *= self.s
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output


class ExpertBranch(nn.Module):
    """
    Individual expert branch - each one is a complete CCNet-like model
    """
    def __init__(self, num_classes, weight, branch_id=0):
        super(ExpertBranch, self).__init__()
        
        self.branch_id = branch_id
        self.num_classes = num_classes
        
        # Each branch has its own set of competitive blocks with slightly different configurations
        # to encourage diversity
        if branch_id == 0:  # Large scale expert
            self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1, weight=weight)
            self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24, weight=weight)
            self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25, weight=weight)
        elif branch_id == 1:  # Medium scale expert
            self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=12, ksize=31, stride=3, padding=15, init_ratio=0.8, weight=weight)
            self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=32, ksize=15, stride=3, padding=7, init_ratio=0.6, o2=20, weight=weight)
            self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=8, ksize=9, stride=3, padding=4, init_ratio=0.3, weight=weight)
        else:  # Small scale expert (branch_id == 2)
            self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=15, ksize=27, stride=3, padding=13, init_ratio=0.6, weight=weight)
            self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=28, ksize=13, stride=3, padding=6, init_ratio=0.7, o2=16, weight=weight)
            self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=10, ksize=11, stride=3, padding=5, init_ratio=0.4, weight=weight)
        
        # Feature dimensions need to be calculated based on the competitive blocks
        # This is a simplified calculation - in practice you'd need to compute this exactly
        feature_dim = 13152  # This should be calculated based on your specific architecture
        
        self.fc = nn.Linear(feature_dim, 4096)
        self.fc1 = nn.Linear(4096, 2048)
        self.drop = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(2048, num_classes)  # Final classifier for this expert
        
    def forward(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.fc(x)
        features = self.fc1(x)
        features = self.drop(features)
        
        # Get logits from this expert
        logits = self.classifier(features)
        
        return logits, features


class EnhancedCCNet(nn.Module):
    """
    Enhanced CCNet with Multi-Expert Architecture
    """
    def __init__(self, num_classes, weight=0.8, num_experts=3):
        super(EnhancedCCNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        # Create expert branches
        self.experts = nn.ModuleList([
            ExpertBranch(num_classes, weight, branch_id=i) 
            for i in range(num_experts)
        ])
        
    def forward(self, x):
        expert_logits = []
        expert_features = []
        
        # Get predictions from each expert
        for expert in self.experts:
            logits, features = expert(x)
            expert_logits.append(logits)
            expert_features.append(features)
        
        # Stack for easier manipulation
        expert_logits = torch.stack(expert_logits, dim=1)  # [batch_size, num_experts, num_classes]
        expert_features = torch.stack(expert_features, dim=1)  # [batch_size, num_experts, feature_dim]
        
        return expert_logits, expert_features
    
    def get_ensemble_prediction(self, expert_logits):
        """
        Get ensemble prediction by averaging expert logits
        """
        return torch.mean(expert_logits, dim=1)
    
    def get_expert_weights(self):
        """
        Extract classifier weights from each expert for diversity loss calculation
        """
        weights = []
        for expert in self.experts:
            weights.append(expert.classifier.weight)
        return weights


# Test the model
if __name__ == "__main__":
    # Test with sample input
    model = EnhancedCCNet(num_classes=600, weight=0.8, num_experts=3)
    x = torch.randn(32, 1, 128, 128)  # batch_size=32, channels=1, height=128, width=128
    
    expert_logits, expert_features = model(x)
    ensemble_pred = model.get_ensemble_prediction(expert_logits)
    
    print(f"Expert logits shape: {expert_logits.shape}")
    print(f"Expert features shape: {expert_features.shape}")
    print(f"Ensemble prediction shape: {ensemble_pred.shape}")
    print("Model created successfully!")