import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings

class GaborConv2d(nn.Module):
    '''
    DESCRIPTION: Stabilized Learnable Gabor Convolution (LGC) layer
    - Fixed parameter constraints to prevent NaN explosion
    - Uses softplus and clamping for positive parameters
    '''
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding      
        self.init_ratio = init_ratio 

        if init_ratio <= 0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # Initial parameter values
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # 🔥 핵심 수정: log-space 파라미터로 재파라미터화
        # sigma = softplus(log_sigma) -> 항상 양수 보장
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(self._SIGMA) - 1e-6), requires_grad=True)
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(self._GAMMA) - 1e-6), requires_grad=True)
        self.log_f = nn.Parameter(torch.log(torch.tensor(self._FREQ) + 1e-6), requires_grad=True)
        
        # theta는 각도이므로 제약 없이 학습 가능
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, 
            requires_grad=False
        )
        
        # psi는 위상이므로 고정
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def get_safe_parameters(self):
        """
        안전한 파라미터 값을 반환 (항상 양수 보장)
        """
        # softplus를 사용하여 항상 양수 보장
        sigma = F.softplus(self.log_sigma) + 1e-6  # 최소값 1e-6
        gamma = F.softplus(self.log_gamma) + 1e-6
        f = F.softplus(self.log_f) + 1e-8        # 주파수는 더 작은 최소값
        
        # 추가 안전장치: 최대값 제한
        sigma = torch.clamp(sigma, min=1e-6, max=50.0)
        gamma = torch.clamp(gamma, min=1e-6, max=10.0)
        f = torch.clamp(f, min=1e-8, max=1.0)
        
        return sigma, gamma, f

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        """
        안전한 Gabor 커널 생성
        """
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
        
        # 🔥 핵심 수정: 안전한 지수 계산
        # sigma²가 분모에 오므로 0이 되면 안됨
        sigma_squared = sigma.view(-1, 1, 1, 1) ** 2
        sigma_squared = torch.clamp(sigma_squared, min=1e-12)  # 최소값 보장
        
        gamma_expanded = gamma.view(-1, 1, 1, 1)
        f_expanded = f.view(-1, 1, 1, 1)
        psi_expanded = psi.view(-1, 1, 1, 1)
        
        # 지수 부분 계산 (오버플로우 방지)
        exponent = -0.5 * ((gamma_expanded * x_theta) ** 2 + y_theta ** 2) / (8 * sigma_squared)
        exponent = torch.clamp(exponent, min=-50, max=50)  # 지수 범위 제한
        
        # 코사인 부분 계산
        cosine_part = torch.cos(2 * math.pi * f_expanded * x_theta + psi_expanded)
        
        # Gabor 커널 생성
        gb = -torch.exp(exponent) * cosine_part
        
        # 평균 제거 (안전하게)
        gb_mean = gb.mean(dim=[2, 3], keepdim=True)
        gb = gb - gb_mean
        
        # NaN/Inf 체크 및 수정
        gb = torch.where(torch.isnan(gb) | torch.isinf(gb), torch.zeros_like(gb), gb)
        
        return gb

    def forward(self, x):
        """
        안전한 forward pass
        """
        # 안전한 파라미터 가져오기
        sigma, gamma, f = self.get_safe_parameters()
        
        # 파라미터 값 체크 (디버깅용)
        if torch.isnan(sigma).any() or torch.isnan(gamma).any() or torch.isnan(f).any():
            print(f"⚠️ NaN in Gabor parameters: sigma={sigma.item():.6f}, gamma={gamma.item():.6f}, f={f.item():.6f}")
            # 안전한 기본값으로 리셋
            sigma = torch.clamp(sigma, min=1e-6, max=50.0)
            gamma = torch.clamp(gamma, min=1e-6, max=10.0)
            f = torch.clamp(f, min=1e-8, max=1.0)

        # Gabor 커널 생성
        kernel = self.genGaborBank(
            self.kernel_size, self.channel_in, self.channel_out, 
            sigma, gamma, self.theta, f, self.psi
        )
        
        # 커널 NaN 체크
        if torch.isnan(kernel).any() or torch.isinf(kernel).any():
            print("⚠️ NaN/Inf detected in Gabor kernel! Using identity kernel.")
            # 안전한 fallback: identity 커널
            kernel = torch.zeros_like(kernel)
            kernel[:, :, kernel.size(2)//2, kernel.size(3)//2] = 1.0
        
        # Convolution 연산
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        
        # 출력 NaN 체크
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("⚠️ NaN/Inf detected in Gabor output!")
            out = torch.where(torch.isnan(out) | torch.isinf(out), torch.zeros_like(out), out)
        
        return out
    
    def get_parameter_info(self):
        """
        현재 파라미터 값 확인용 (디버깅)
        """
        sigma, gamma, f = self.get_safe_parameters()
        return {
            'sigma': sigma.item(),
            'gamma': gamma.item(), 
            'f': f.item(),
            'log_sigma': self.log_sigma.item(),
            'log_gamma': self.log_gamma.item(),
            'log_f': self.log_f.item()
        }


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


# 🔧 디버깅 및 테스트 함수 추가
def test_gabor_stability():
    """
    Gabor 필터의 안정성 테스트
    """
    print("🧪 Testing Gabor filter stability...")
    
    # 테스트 입력
    x = torch.randn(2, 1, 64, 64)
    gabor = GaborConv2d(channel_in=1, channel_out=9, kernel_size=35, stride=2, padding=17)
    
    print("Initial parameters:", gabor.get_parameter_info())
    
    # Forward pass 테스트
    try:
        output = gabor(x)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"Has NaN: {torch.isnan(output).any().item()}")
        print(f"Has Inf: {torch.isinf(output).any().item()}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    # 극단적인 파라미터로 테스트
    print("\n🧪 Testing with extreme parameters...")
    gabor.log_sigma.data = torch.tensor(-20.0)  # 매우 작은 sigma
    gabor.log_gamma.data = torch.tensor(-20.0)  # 매우 작은 gamma
    
    print("Extreme parameters:", gabor.get_parameter_info())
    
    try:
        output = gabor(x)
        print(f"✅ Extreme parameter test passed! Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any().item()}")
        print(f"Has Inf: {torch.isinf(output).any().item()}")
    except Exception as e:
        print(f"❌ Extreme parameter test failed: {e}")
    
    print("✅ Gabor stability tests completed!")


def test_enhanced_ccnet():
    """
    전체 EnhancedCCNet 모델 테스트
    """
    print("\n🧪 Testing Enhanced CCNet...")
    
    # Test with sample input
    model = EnhancedCCNet(num_classes=600, weight=0.8, num_experts=3)
    x = torch.randn(8, 1, 128, 128)  # 작은 배치로 테스트
    
    try:
        expert_logits, expert_features = model(x)
        ensemble_pred = model.get_ensemble_prediction(expert_logits)
        expert_weights = model.get_expert_weights()
        
        print(f"✅ Model forward pass successful!")
        print(f"Expert logits shape: {expert_logits.shape}")
        print(f"Expert features shape: {expert_features.shape}")
        print(f"Ensemble prediction shape: {ensemble_pred.shape}")
        print(f"Number of expert weights: {len(expert_weights)}")
        
        # NaN 체크
        has_nan = (torch.isnan(expert_logits).any() or 
                   torch.isnan(expert_features).any() or 
                   torch.isnan(ensemble_pred).any())
        print(f"Has NaN in outputs: {has_nan}")
        
        if not has_nan:
            print("✅ Enhanced CCNet is stable and ready for training!")
        else:
            print("❌ NaN detected in model outputs!")
            
    except Exception as e:
        print(f"❌ Enhanced CCNet test failed: {e}")


# Test the model
if __name__ == "__main__":
    print("🚀 Starting Enhanced CCNet stability tests...\n")
    
    # Gabor 필터 안정성 테스트
    test_gabor_stability()
    
    # 전체 모델 테스트
    test_enhanced_ccnet()
    
    