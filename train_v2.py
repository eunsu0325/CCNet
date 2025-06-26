import torch.nn.utils as utils  # 맨 위 import 부분에 추가
import os
import argparse
import time
import sys
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import cv2 as cv

# Import our enhanced modules
from models.ccnet_v2 import EnhancedCCNet
from loss_v2 import EnhancedCCNetLoss, ContrastiveLoss
from models import MyDataset
from utils import *

def safe_training_step(model, optimizer, loss, max_grad_norm=0.5):
    """
    완전히 안전한 훈련 스텝
    """
    # Backward
    loss.backward()
    
    # 1. 그래디언트 NaN 체크
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"⚠️ NaN/Inf gradient in {name}")
                has_nan_grad = True
                param.grad.zero_()  # NaN 그래디언트 제거
    
    if has_nan_grad:
        print("⚠️ Skipping optimizer step due to NaN gradients")
        optimizer.zero_grad()
        return False
    
    # 2. 그래디언트 클리핑 (매우 보수적)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # 3. 옵티마이저 스텝
    optimizer.step()
    optimizer.zero_grad()
    
    # 4. 파라미터 NaN 체크 및 복구
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            print(f"⚠️ NaN/Inf parameter in {name}, reinitializing...")
            if 'sigma_raw' in name:
                param.data = torch.tensor(9.2).to(param.device)
            elif 'gamma_raw' in name:
                param.data = torch.tensor(2.0).to(param.device)
            elif 'f_raw' in name:
                param.data = torch.tensor(0.057).to(param.device)
            else:
                param.data.normal_(0, 0.01)  # 다른 파라미터는 작은 노이즈로
    
    return True

def test_enhanced_model(model, train_set_file, test_set_file, path_rst):
    """
    Enhanced testing function for multi-expert model
    """
    print('Start Testing Enhanced CCNet!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(path_rst, 'rank1_hard')

    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    batch_size = 512
    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    if not os.path.exists(path_rst):
        os.makedirs(path_rst)
    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    model.cuda()
    model.eval()

    # Feature extraction for training set
    featDB_train = []
    iddb_train = []

    print('Extracting features from training set...')
    for batch_id, (datas, target) in enumerate(data_loader_train):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        with torch.no_grad():
            expert_logits, expert_features = model(data)
            # Use ensemble prediction for feature extraction
            ensemble_logits = model.get_ensemble_prediction(expert_logits)
            # Use ensemble features (average of expert features)
            ensemble_features = torch.mean(expert_features, dim=1)  # [batch_size, feature_dim]
            
        codes = F.normalize(ensemble_features, p=2, dim=1)  # L2 normalize
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]
    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)

    # Feature extraction for test set
    featDB_test = []
    iddb_test = []

    print('Extracting features from test set...')
    for batch_id, (datas, target) in enumerate(data_loader_test):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        with torch.no_grad():
            expert_logits, expert_features = model(data)
            ensemble_logits = model.get_ensemble_prediction(expert_logits)
            ensemble_features = torch.mean(expert_features, dim=1)

        codes = F.normalize(ensemble_features, p=2, dim=1)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    print('completed feature extraction for test set.')
    print('featDB_test.shape: ', featDB_test.shape)
    print('\nFeature extraction done!')

    # Verification EER calculation
    print('Calculating verification EER...')
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]
        for j in range(ntrain):
            feat2 = featDB_train[j]
            
            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(path_rst+'veriEER'):
        os.makedirs(path_rst+'veriEER')

    with open(path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    # Rank-1 accuracy calculation
    print('\nCalculating Rank-1 accuracy...')
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]
        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])
        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            if len(fileDB_test) > i and len(fileDB_train) > idx:
                testname = fileDB_test[i]
                trainname = fileDB_train[idx]
                # Store similar inter-class samples
                try:
                    im_test = cv.imread(testname)
                    im_train = cv.imread(trainname)
                    if im_test is not None and im_train is not None:
                        img = np.concatenate((im_test, im_train), axis=1)
                        cv.imwrite(path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                            np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)
                except:
                    pass  # Skip if image loading fails

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    # Test set EER calculation
    print('\nCalculating test set EER...')
    s = []
    l = []
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]
        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            s.append(dis)

            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    with open(path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')


def fit_enhanced_model(epoch, model, data_loader, criterion, optimizer, phase='training'):
    """
    Enhanced training function for multi-expert model
    """
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_losses = {
        'total': 0.0,
        'base': 0.0,
        'okd': 0.0,
        'cd': 0.0,
        'contrastive': 0.0
    }
    running_correct = 0
    total_samples = 0

    for batch_id, (datas, target) in enumerate(data_loader):
        data = datas[0]
        data_con = datas[1]  # Contrastive sample
        
        data = data.cuda()
        data_con = data_con.cuda()
        target = target.cuda()

        if phase == 'training':
            optimizer.zero_grad()

        # Forward pass through enhanced model
        if phase == 'training':
            expert_logits, expert_features = model(data)
            expert_logits_con, expert_features_con = model(data_con)
        else:
            with torch.no_grad():
                expert_logits, expert_features = model(data)
                expert_logits_con, expert_features_con = model(data_con)

        # Get expert weights for diversity loss
        expert_weights = model.get_expert_weights()

        # Calculate main losses (base + OKD + CD)
        loss_dict = criterion(expert_logits, expert_weights, target)
        
        # Calculate contrastive loss if we have contrastive samples
        contrastive_loss = 0.0
        if data_con is not None:
            # Combine features from both samples for contrastive learning
            combined_features = torch.cat([
                expert_features.unsqueeze(2), 
                expert_features_con.unsqueeze(2)
            ], dim=2)  # [batch_size, num_experts, 2, feature_dim]
            
            # Average across experts for contrastive loss
            avg_features = torch.mean(combined_features, dim=1)  # [batch_size, 2, feature_dim]
            
            # Apply contrastive loss (you might want to adjust this)
            # contrastive_criterion = ContrastiveLoss(temperature=0.07)
            # contrastive_loss = contrastive_criterion(avg_features, target)

        # Total loss
        total_loss = loss_dict['total_loss'] + 0.2 * contrastive_loss

        # Get ensemble prediction for accuracy calculation
        ensemble_logits = model.get_ensemble_prediction(expert_logits)
        preds = ensemble_logits.data.max(dim=1, keepdim=True)[1]
        correct = preds.eq(target.data.view_as(preds)).cpu().sum().numpy()

        # Update running statistics
        batch_size = data.size(0)
        total_samples += batch_size
        running_correct += correct
        running_losses['total'] += total_loss.item() * batch_size
        running_losses['base'] += loss_dict['base_loss'].item() * batch_size
        running_losses['okd'] += loss_dict['okd_loss'].item() * batch_size
        running_losses['cd'] += loss_dict['cd_loss'].item() * batch_size
        running_losses['contrastive'] += contrastive_loss * batch_size if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss * batch_size

        # Backward pass
        if phase == 'training':
            success = safe_training_step(model, optimizer, total_loss, max_grad_norm=0.5)
            if not success:
                print(f"⚠️ Training step failed at epoch {epoch}, batch {batch_id}")
                continue  # 이 배치 스킵


    # Calculate averages
    avg_losses = {k: v / total_samples for k, v in running_losses.items()}
    accuracy = (100.0 * running_correct) / total_samples

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: {phase}')
        print(f'  Total Loss: {avg_losses["total"]:.5f}')
        print(f'  Base Loss: {avg_losses["base"]:.5f}')
        print(f'  OKD Loss: {avg_losses["okd"]:.5f}')
        print(f'  CD Loss: {avg_losses["cd"]:.8f}')
        print(f'  Contrastive Loss: {avg_losses["contrastive"]:.5f}')
        print(f'  Accuracy: {running_correct}/{total_samples} ({accuracy:.3f}%)')

    return avg_losses, accuracy


def main():
    parser = argparse.ArgumentParser(description="Enhanced CCNet for Palmprint Recognition")
    
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch_num", type=int, default=3000)
    parser.add_argument("--lambda_okd", type=float, default=1.0)
    parser.add_argument("--lambda_cd", type=float, default=5e-8)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--comp_weight", type=float, default=0.8)
    parser.add_argument("--id_num", type=int, default=378, 
                        help="Number of classes: IITD: 460, Tongji: 600, PolyU: 378, etc.")
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--redstep", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=500)
    
    # Paths
    parser.add_argument("--train_set_file", type=str, default='./data/train_Tongji.txt')
    parser.add_argument("--test_set_file", type=str, default='./data/test_Tongji.txt')
    parser.add_argument("--des_path", type=str, default='./results/enhanced_checkpoint/')
    parser.add_argument("--path_rst", type=str, default='./results/enhanced_rst_test/')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Create directories
    if not os.path.exists(args.des_path):
        os.makedirs(args.des_path)
    if not os.path.exists(args.path_rst):
        os.makedirs(args.path_rst)
    
    print('Enhanced CCNet Training Configuration:')
    print(f'  Number of experts: {args.num_experts}')
    print(f'  Lambda OKD: {args.lambda_okd}')
    print(f'  Lambda CD: {args.lambda_cd}')
    print(f'  Temperature: {args.temperature}')
    print(f'  Competition weight: {args.comp_weight}')
    
    # Dataset
    trainset = MyDataset(txt=args.train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    testset = MyDataset(txt=args.test_set_file, transforms=None, train=False, imside=128, outchannels=1)
    
    data_loader_train = DataLoader(dataset=trainset, batch_size=args.batch_size, num_workers=2, shuffle=True)
    data_loader_test = DataLoader(dataset=testset, batch_size=128, num_workers=2, shuffle=True)
    
    print(f'Training started at: {time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}')
    
    # Model
    print('Initializing Enhanced CCNet...')
    model = EnhancedCCNet(
        num_classes=args.id_num,
        weight=args.comp_weight,
        num_experts=args.num_experts
    )
    best_model = copy.deepcopy(model)
    model.cuda()
    
    # Loss function
    criterion = EnhancedCCNetLoss(
        lambda_okd=args.lambda_okd,
        lambda_cd=args.lambda_cd,
        temperature=args.temperature
    )
    
    # Optimizer and scheduler
    gabor_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'logit' in name:  # Gabor의 logit 파라미터들
            gabor_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.lr},
        {'params': gabor_params, 'lr': args.lr * 0.01}  # Gabor는 1% 학습률
    ], weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.redstep, gamma=0.8)
    
    # Training loop
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    best_acc = 0
    
    for epoch in range(args.epoch_num):
        # Training
        epoch_losses, epoch_accuracy = fit_enhanced_model(
            epoch, model, data_loader_train, criterion, optimizer, phase='training'
        )
        
        # Validation
        val_epoch_losses, val_epoch_accuracy = fit_enhanced_model(
            epoch, model, data_loader_test, criterion, None, phase='testing'
        )
        
        scheduler.step()
        
        # Log results
        train_losses.append(epoch_losses['total'])
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_losses['total'])
        val_accuracy.append(val_epoch_accuracy)
        
        # Save best model
        if epoch_accuracy >= best_acc:
            best_acc = epoch_accuracy
            torch.save(model.state_dict(), args.des_path + 'enhanced_net_params_best.pth')
            best_model = copy.deepcopy(model)
        
        # Save current model
        if epoch % 10 == 0 or epoch == (args.epoch_num - 1):
            torch.save(model.state_dict(), args.des_path + 'enhanced_net_params.pth')
            saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, best_acc, args.path_rst)
        
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), args.des_path + f'enhanced_epoch_{epoch}_net_params.pth')
        
        if epoch % args.test_interval == 0 and epoch != 0:
            print('Running intermediate test...')
            test_enhanced_model(model, args.train_set_file, args.test_set_file, args.path_rst)
    
    print('Final testing with last model...')
    test_enhanced_model(model, args.train_set_file, args.test_set_file, args.path_rst)
    
    print('Final testing with best model...')
    test_enhanced_model(best_model, args.train_set_file, args.test_set_file, args.path_rst)
    
    print(f'Training completed at: {time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}')
    print(f'Best accuracy achieved: {best_acc:.3f}%')


if __name__ == "__main__":
    main()
