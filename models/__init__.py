# models/__init__.py 수정
from models.ccnet import ccnet
from models.enhanced_ccnet import EnhancedCCNet
from models.dataset import MyDataset
from models.dataset import NormSingleROI

# enhanced_ccnet.py 의 import 수정을 위한 패치
"""
enhanced_ccnet.py 파일 상단에 다음을 추가:

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

# 기존 ccnet과의 호환성을 위한 래퍼 클래스
class CCNetCompatibilityWrapper:
    """
    기존 ccnet 인터페이스와 호환되는 래퍼 클래스
    """
    def __init__(self, enhanced_model):
        self.enhanced_model = enhanced_model
        self.num_classes = enhanced_model.num_classes
        
    def forward(self, x, y=None):
        """
        기존 ccnet과 동일한 인터페이스 제공
        """
        expert_logits, expert_features = self.enhanced_model(x)
        ensemble_logits = self.enhanced_model.get_ensemble_prediction(expert_logits)
        ensemble_features = torch.mean(expert_features, dim=1)
        
        # 기존 ccnet과 동일한 형태로 반환
        return ensemble_logits, F.normalize(ensemble_features, dim=-1)
    
    def getFeatureCode(self, x):
        """
        기존 inference.py와 호환되는 특징 추출 함수
        """
        expert_logits, expert_features = self.enhanced_model(x)
        ensemble_features = torch.mean(expert_features, dim=1)
        normalized_features = F.normalize(ensemble_features, p=2, dim=1)
        return normalized_features
    
    def cuda(self):
        self.enhanced_model.cuda()
        return self
    
    def eval(self):
        self.enhanced_model.eval()
        return self
    
    def train(self):
        self.enhanced_model.train()
        return self
    
    def state_dict(self):
        return self.enhanced_model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        return self.enhanced_model.load_state_dict(state_dict, strict)
    
    def parameters(self):
        return self.enhanced_model.parameters()


# 기존 inference.py 를 최소한으로 수정하여 Enhanced CCNet 사용
def create_enhanced_ccnet_for_inference(num_classes, weight=0.8, num_experts=3):
    """
    기존 inference 스크립트에서 사용할 수 있는 Enhanced CCNet 생성
    """
    enhanced_model = EnhancedCCNet(num_classes=num_classes, weight=weight, num_experts=num_experts)
    wrapper = CCNetCompatibilityWrapper(enhanced_model)
    return wrapper


# 하이퍼파라미터 추천값들
RECOMMENDED_HYPERPARAMETERS = {
    "Tongji": {
        "lambda_okd": 1.0,
        "lambda_cd": 5e-8,
        "temperature": 3.0,
        "comp_weight": 0.8,
        "lr": 0.001,
        "batch_size": 1024
    },
    "PolyU": {
        "lambda_okd": 0.8,
        "lambda_cd": 1e-7,
        "temperature": 4.0,
        "comp_weight": 0.7,
        "lr": 0.001,
        "batch_size": 512
    },
    "IITD": {
        "lambda_okd": 1.2,
        "lambda_cd": 3e-8,
        "temperature": 3.5,
        "comp_weight": 0.8,
        "lr": 0.0008,
        "batch_size": 1024
    },
    "Multi-Spectrum": {
        "lambda_okd": 0.9,
        "lambda_cd": 8e-8,
        "temperature": 2.5,
        "comp_weight": 0.9,
        "lr": 0.001,
        "batch_size": 768
    }
}

def get_recommended_hyperparameters(dataset_name):
    """
    데이터셋에 따른 추천 하이퍼파라미터 반환
    """
    return RECOMMENDED_HYPERPARAMETERS.get(dataset_name, RECOMMENDED_HYPERPARAMETERS["Tongji"])


# 기존 train.py에서 Enhanced CCNet을 사용하기 위한 최소 수정사항
def patch_existing_train_script():
    """
    기존 train.py를 Enhanced CCNet으로 교체하기 위한 패치 가이드
    
    1. import 추가:
       from enhanced_ccnet import EnhancedCCNet
       from enhanced_loss_functions import EnhancedCCNetLoss
    
    2. 모델 생성 부분 교체:
       기존: net = ccnet(num_classes=num_classes, weight=comp_weight)
       변경: net = EnhancedCCNet(num_classes=num_classes, weight=comp_weight, num_experts=3)
    
    3. 손실 함수 교체:
       기존: criterion = nn.CrossEntropyLoss()
       변경: criterion = EnhancedCCNetLoss(lambda_okd=1.0, lambda_cd=5e-8, temperature=3.0)
    
    4. forward pass 수정:
       기존: output, fe1 = model(data, target)
       변경: expert_logits, expert_features = model(data)
             ensemble_output = model.get_ensemble_prediction(expert_logits)
             expert_weights = model.get_expert_weights()
             loss_dict = criterion(expert_logits, expert_weights, target)
    """
    pass


# 디버깅 및 검증을 위한 유틸리티 함수들
def validate_model_output(model, sample_input):
    """
    모델 출력 형태 검증
    """
    model.eval()
    with torch.no_grad():
        try:
            expert_logits, expert_features = model(sample_input)
            ensemble_pred = model.get_ensemble_prediction(expert_logits)
            expert_weights = model.get_expert_weights()
            
            print("Model validation successful!")
            print(f"Expert logits shape: {expert_logits.shape}")
            print(f"Expert features shape: {expert_features.shape}")
            print(f"Ensemble prediction shape: {ensemble_pred.shape}")
            print(f"Number of expert weights: {len(expert_weights)}")
            
            return True
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False


def check_loss_computation(model, sample_input, sample_target):
    """
    손실 함수 계산 검증
    """
    from enhanced_loss_functions import EnhancedCCNetLoss
    
    model.train()
    criterion = EnhancedCCNetLoss()
    
    try:
        expert_logits, expert_features = model(sample_input)
        expert_weights = model.get_expert_weights()
        loss_dict = criterion(expert_logits, expert_weights, sample_target)
        
        print("Loss computation successful!")
        print(f"Total loss: {loss_dict['total_loss'].item():.6f}")
        print(f"Base loss: {loss_dict['base_loss'].item():.6f}")
        print(f"OKD loss: {loss_dict['okd_loss'].item():.6f}")
        print(f"CD loss: {loss_dict['cd_loss'].item():.8f}")
        
        return True
    except Exception as e:
        print(f"Loss computation failed: {e}")
        return False


# 메모리 사용량 최적화를 위한 팁들
def optimize_memory_usage():
    """
    메모리 사용량 최적화 가이드
    
    1. 그래디언트 체크포인팅 사용:
       from torch.utils.checkpoint import checkpoint
       
    2. 혼합 정밀도 훈련:
       from torch.cuda.amp import autocast, GradScaler
       
    3. 배치 크기 조정:
       - 3 experts일 때: batch_size를 기존의 2/3로 설정
       - GPU 메모리에 따라 동적 조정
    
    4. 특징 캐싱:
       - 훈련 중 일부 특징을 디스크에 캐시
    """
    pass


# 마이그레이션 체크리스트
MIGRATION_CHECKLIST = """
Enhanced CCNet 마이그레이션 체크리스트:

□ 1. enhanced_ccnet.py 파일 추가
□ 2. enhanced_loss_functions.py 파일 추가  
□ 3. models/__init__.py에 import 추가
□ 4. 기존 데이터셋 파일들 확인 (train/test txt files)
□ 5. GPU 메모리 충분한지 확인 (기존 대비 약 1.5~2배 필요)
□ 6. 하이퍼파라미터 설정 확인
□ 7. 샘플 입력으로 모델 검증
□ 8. 손실 함수 계산 검증
□ 9. 작은 데이터셋으로 훈련 테스트
□ 10. 전체 데이터셋으로 훈련 시작

문제 발생 시 체크사항:
- CUDA 메모리 부족: 배치 크기 줄이기
- 수렴 안됨: lambda_okd 값 조정
- 다양성 부족: lambda_cd 값 증가
- 너무 느림: num_experts 줄이기
"""


if __name__ == "__main__":
    print("Enhanced CCNet 호환성 검증 시작...")
    
    # 샘플 데이터로 검증
    sample_input = torch.randn(8, 1, 128, 128)
    sample_target = torch.randint(0, 600, (8,))
    
    # 모델 생성 및 검증
    model = EnhancedCCNet(num_classes=600, weight=0.8, num_experts=3)
    
    print("\n1. 모델 출력 검증:")
    validate_model_output(model, sample_input)
    
    print("\n2. 손실 함수 검증:")
    check_loss_computation(model, sample_input, sample_target)
    
    print("\n3. 호환성 래퍼 테스트:")
    wrapper = CCNetCompatibilityWrapper(model)
    try:
        output, features = wrapper.forward(sample_input)
        feature_codes = wrapper.getFeatureCode(sample_input)
        print("호환성 래퍼 테스트 성공!")
        print(f"Output shape: {output.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Feature codes shape: {feature_codes.shape}")
    except Exception as e:
        print(f"호환성 래퍼 테스트 실패: {e}")
    
    print("\n" + MIGRATION_CHECKLIST)