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

        # assert channel_in == 1

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding      

        self.init_ratio = init_ratio 

        self.kernel = 0

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

        # x=x.float()
        # y=y.float()

        # Rotated coordinate systems
        # [channel_out, <channel_in, kernel, kernel>], broadcasting
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))  
                
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
    
        gb = gb - gb.mean(dim=[2,3], keepdim=True)

        return gb


    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        # print(x.shape)
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
    DESCRIPTION: an implementation of the Competitive Block::

    [CB = LGC + argmax + PPU] \n

    INPUTS: \n

    channel_in: only support 1 \n
    n_competitor: number of channels of the LGC (channel_out)  \n

    ksize, stride, padding: 2D convolution parameters \n

    init_ratio: scale factor of the initial parameters (receptive filed) \n

    o1, o2: numbers of channels of the conv_1 and conv_2 layers in the PPU, respectively. (PPU parameters)
    '''

    def __init__(self, channel_in, n_competitor, ksize, stride, padding,weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp, self).__init__()

        # assert channel_in == 1
        self.channel_in = channel_in
        self.n_competitor = n_competitor

        self.init_ratio = init_ratio

        self.gabor_conv2d = GaborConv2d(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                        padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                         padding=ksize // 2, init_ratio=init_ratio)
        ## 2 2 no conv layer
        # soft-argmax
        # self.a = nn.Parameter(torch.FloatTensor([1]))
        # self.b = nn.Parameter(torch.FloatTensor([0]))

        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        # PPU
        self.conv1_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight
        self.weight_spa = (1-weight) / 2
        # print(self.weight_chan)
    def forward(self, x):

        #1-st order
        x = self.gabor_conv2d(x)
        # print(x.shape)
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

        xx = torch.cat((x_1.view(x_1.shape[0],-1),x_2.view(x_2.shape[0],-1)),dim=1)

        return xx


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance::
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)

        From: https://github.com/ronghuaiyang/arcface-pytorch
        """
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
        if self.training :
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
            # assert label is None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output


# 🆕 Multi-Scale Fusion 클래스 추가
class BasicMultiScaleFusion(nn.Module):
    """
    CCNet용 기본 Multi-Scale Fusion
    128×128 입력, CB 출력 차원: 4384, 4384, 4384
    """
    def __init__(self, cb1_dim=4384, cb2_dim=4384, cb3_dim=4384):
        super().__init__()
        
        self.cb1_dim = cb1_dim
        self.cb2_dim = cb2_dim  
        self.cb3_dim = cb3_dim
        total_dim = cb1_dim + cb2_dim + cb3_dim  # 13152
        
        print(f"🔧 Multi-Scale Fusion 초기화: CB1={cb1_dim}, CB2={cb2_dim}, CB3={cb3_dim}, Total={total_dim}")
        
        # 스케일별 중요도 예측 네트워크
        self.scale_importance_net = nn.Sequential(
            nn.Linear(total_dim, 512),           # 13152 → 512
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),                 # 512 → 128
            nn.ReLU(),
            nn.Linear(128, 3),                   # 128 → 3 (Large, Medium, Small)
            nn.Softmax(dim=1)
        )
        
        # 품질 추정기 (전체 특징의 품질 평가)
        self.quality_estimator = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 0~1 사이 품질 점수
        )
        
    def forward(self, x1, x2, x3, return_weights=False):
        """
        x1: CB1 출력 features [Batch, 4384] 
        x2: CB2 출력 features [Batch, 4384]
        x3: CB3 출력 features [Batch, 4384]
        """
        batch_size = x1.size(0)
        
        # Step 1: 모든 스케일 특징 결합
        all_features = torch.cat([x1, x2, x3], dim=1)  # [Batch, 13152]
        
        # Step 2: 스케일별 중요도 계산
        scale_weights = self.scale_importance_net(all_features)  # [Batch, 3]
        
        # Step 3: 품질 추정
        feature_quality = self.quality_estimator(all_features)  # [Batch, 1]
        
        # Step 4: 가중치 적용
        weighted_x1 = x1 * scale_weights[:, 0:1]  # Large scale 가중치
        weighted_x2 = x2 * scale_weights[:, 1:2]  # Medium scale 가중치
        weighted_x3 = x3 * scale_weights[:, 2:3]  # Small scale 가중치
        
        # Step 5: 가중 결합
        fused_features = torch.cat([weighted_x1, weighted_x2, weighted_x3], dim=1)  # [Batch, 13152]
        
        if return_weights:
            return fused_features, {
                'scale_weights': scale_weights,
                'feature_quality': feature_quality,
                'original_concat': all_features
            }
        return fused_features


# 🔧 수정된 CCNet 클래스
class ccnet(torch.nn.Module):
    '''
    Enhanced CCNet with Multi-Scale Fusion
    기존 CCNet + Multi-Scale Fusion 적용
    '''

    def __init__(self, num_classes, weight):
        super(ccnet, self).__init__()

        self.num_classes = num_classes

        print(f"🚀 CCNet 초기화: num_classes={num_classes}, competition_weight={weight}")

        # 기존 competitive blocks (변경 없음)
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1, weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24, weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25, weight=weight)

        # 🆕 Multi-Scale Fusion 추가
        self.multi_scale_fusion = BasicMultiScaleFusion(
            cb1_dim=4384,  # 코랩 분석 결과
            cb2_dim=4384,  # 코랩 분석 결과  
            cb3_dim=4384   # 코랩 분석 결과
        )
        
        # 기존 FC layers (변경 없음)
        self.fc = torch.nn.Linear(13152, 4096)  # 13152 = 4384 + 4384 + 4384
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

        print("✅ CCNet 초기화 완료!")

    def forward(self, x, y=None, return_analysis=False):
        # 기존 feature extraction (변경 없음)
        x1 = self.cb1(x)  # [Batch, 4384]
        x2 = self.cb2(x)  # [Batch, 4384]
        x3 = self.cb3(x)  # [Batch, 4384]

        # 🆕 Multi-Scale Fusion 적용
        if return_analysis:
            fused_x, analysis_info = self.multi_scale_fusion(x1, x2, x3, return_weights=True)
        else:
            fused_x = self.multi_scale_fusion(x1, x2, x3)

        # 나머지는 기존과 동일 (변경 없음)
        x1_fc = self.fc(fused_x)      # [Batch, 4096]
        x_fc = self.fc1(x1_fc)        # [Batch, 2048]
        fe = torch.cat((x1_fc, x_fc), dim=1)  # [Batch, 6144] 특징 벡터
        x_dropped = self.drop(x_fc)   # [Batch, 2048] 
        output = self.arclayer_(x_dropped, y)  # [Batch, num_classes]

        if return_analysis:
            return output, F.normalize(fe, dim=-1), analysis_info
        
        return output, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        # 기존 feature extraction (변경 없음)
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        # 🆕 Multi-Scale Fusion 적용
        fused_x = self.multi_scale_fusion(x1, x2, x3)

        # 나머지는 기존과 동일 (변경 없음)
        x = self.fc(fused_x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x

    def analyze_fusion_weights(self, x):
        """Multi-Scale Fusion 가중치 분석용 함수"""
        self.eval()
        with torch.no_grad():
            x1 = self.cb1(x)
            x2 = self.cb2(x)
            x3 = self.cb3(x)
            
            _, analysis_info = self.multi_scale_fusion(x1, x2, x3, return_weights=True)
            
            return analysis_info


if __name__ == "__main__":
    print("🧪 Enhanced CCNet with Multi-Scale Fusion 테스트")
    print("="*60)
    
    # 테스트 입력 생성
    dummy_input = torch.randn(4, 1, 128, 128)
    print(f"📥 테스트 입력 shape: {dummy_input.shape}")
    
    # 모델 생성
    net = ccnet(num_classes=600, weight=0.8)
    net.eval()
    
    print("\n🚀 기본 Forward 테스트:")
    with torch.no_grad():
        output, features = net(dummy_input)
        print(f"  출력 shape: {output.shape}")
        print(f"  특징 벡터 shape: {features.shape}")
    
    print("\n🔍 Multi-Scale Fusion 분석:")
    with torch.no_grad():
        output, features, analysis = net(dummy_input, return_analysis=True)
        scale_weights = analysis['scale_weights']
        feature_quality = analysis['feature_quality']
        
        print(f"  스케일 가중치 평균:")
        print(f"    Large (CB1):  {scale_weights[:, 0].mean():.3f}")
        print(f"    Medium (CB2): {scale_weights[:, 1].mean():.3f}")
        print(f"    Small (CB3):  {scale_weights[:, 2].mean():.3f}")
        print(f"  특징 품질 평균: {feature_quality.mean():.3f}")
    
    print("\n🎯 getFeatureCode 테스트:")
    with torch.no_grad():
        feature_code = net.getFeatureCode(dummy_input)
        print(f"  특징 코드 shape: {feature_code.shape}")
        print(f"  정규화 확인: {torch.norm(feature_code[0]).item():.6f} (1에 가까워야 함)")
    
    print("\n✅ 모든 테스트 통과!")
    print("="*60)
