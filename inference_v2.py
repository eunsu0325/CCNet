import os
import argparse
import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F  # 추가된 import
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models

import pickle
import numpy as np
from PIL import Image
import cv2 as cv

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Import enhanced modules - 수정된 import
from models.ccnet_v2 import EnhancedCCNet  # 올바른 import 경로
from models import MyDataset
from utils import *

import copy
import argparse


def test_enhanced_model(model):
    """
    Enhanced inference function for multi-expert model
    """
    print('Start Enhanced CCNet Inference!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    ### Calculate EER for enhanced model
    path_hard = os.path.join(path_rst, 'rank1_hard')

    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    batch_size = 1024
    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=0)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=0)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    # Output directories
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)
    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model
    net.cuda()
    net.eval()

    # Feature extraction for training set
    print('Extracting features from training set...')
    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        with torch.no_grad():
            expert_logits, expert_features = net(data)
            # Use ensemble features (average of all experts)
            ensemble_features = torch.mean(expert_features, dim=1)  # [batch_size, feature_dim]
            
        # Normalize features
        codes = F.normalize(ensemble_features, p=2, dim=1)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('Completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]
    assert num_training_samples % classNumel == 0
    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print()

    # Feature extraction for test set
    print('Extracting features from test set...')
    featDB_test = []
    iddb_test = []

    for batch_id, (datas, target) in enumerate(data_loader_test):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        with torch.no_grad():
            expert_logits, expert_features = net(data)
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

    print('Completed feature extraction for test set.')
    print('featDB_test.shape: ', featDB_test.shape)
    print('\nFeature extraction done!')
    print()

    # Expert analysis: Show individual expert performance
    print('Analyzing individual expert performance...')
    for expert_idx in range(net.num_experts):
        print(f'\n--- Expert {expert_idx + 1} Analysis ---')
        
        # Get features from individual expert
        expert_featDB_test = []
        for batch_id, (datas, target) in enumerate(data_loader_test):
            data = datas[0]
            data = data.cuda()
            
            with torch.no_grad():
                expert_logits, expert_features = net(data)
                # Use only this expert's features
                expert_feat = expert_features[:, expert_idx, :]  # [batch_size, feature_dim]
                
            codes = F.normalize(expert_feat, p=2, dim=1)
            codes = codes.cpu().detach().numpy()
            
            if batch_id == 0:
                expert_featDB_test = codes
            else:
                expert_featDB_test = np.concatenate((expert_featDB_test, codes), axis=0)
        
        # Calculate accuracy for this expert
        correct = 0
        total = len(iddb_test)
        
        for i in range(total):
            feat_query = expert_featDB_test[i]
            query_id = iddb_test[i]
            
            # Find best match in training set
            best_score = -1
            best_match_id = -1
            
            for j in range(len(featDB_train)):
                score = np.dot(feat_query, featDB_train[j])  # Cosine similarity
                if score > best_score:
                    best_score = score
                    best_match_id = iddb_train[j]
            
            if query_id == best_match_id:
                correct += 1
        
        expert_accuracy = (correct / total) * 100
        print(f'Expert {expert_idx + 1} individual accuracy: {correct}/{total} ({expert_accuracy:.3f}%)')

    print('\n--- Ensemble Performance ---')

    # Verification EER calculation
    print('Verification EER of the test set...')
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
    if not os.path.exists(path_rst+'veriEER/rank1_hard/'):
        os.makedirs(path_rst+'veriEER/rank1_hard/')

    with open(path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
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
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            try:
                im_test = cv.imread(testname)
                im_train = cv.imread(trainname)
                if im_test is not None and im_train is not None:
                    img = np.concatenate((im_test, im_train), axis=1)
                    cv.imwrite(path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                        np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)
            except:
                pass  # Skip if error in image processing

    rankacc = corr / ntest * 100
    print('Ensemble rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('Ensemble rank-1 acc: %.3f%%' % rankacc)

    # Expert voting analysis
    print('\n--- Expert Voting Analysis ---')
    analyze_expert_voting(net, data_loader_test, iddb_test, featDB_train, iddb_train)


def analyze_expert_voting(model, data_loader_test, iddb_test, featDB_train, iddb_train):
    """
    Analyze how experts vote and their agreement
    """
    model.eval()
    
    # Collect all expert predictions
    all_expert_logits = []
    test_targets = []
    
    for batch_id, (datas, target) in enumerate(data_loader_test):
        data = datas[0]
        data = data.cuda()
        
        with torch.no_grad():
            expert_logits, _ = model(data)  # [batch_size, num_experts, num_classes]
        
        all_expert_logits.append(expert_logits.cpu())
        test_targets.append(target)
    
    # Concatenate all batches
    all_expert_logits = torch.cat(all_expert_logits, dim=0)  # [total_test_samples, num_experts, num_classes]
    test_targets = torch.cat(test_targets, dim=0)
    
    # Analyze expert agreement
    num_samples = all_expert_logits.shape[0]
    num_experts = all_expert_logits.shape[1]
    
    agreements = 0
    disagreements = 0
    
    expert_predictions = torch.argmax(all_expert_logits, dim=2)  # [num_samples, num_experts]
    
    for i in range(num_samples):
        expert_preds = expert_predictions[i]  # [num_experts]
        
        # Check if all experts agree
        if torch.all(expert_preds == expert_preds[0]):
            agreements += 1
        else:
            disagreements += 1
    
    agreement_rate = (agreements / num_samples) * 100
    print(f'Expert agreement rate: {agreements}/{num_samples} ({agreement_rate:.2f}%)')
    print(f'Expert disagreement rate: {disagreements}/{num_samples} ({100-agreement_rate:.2f}%)')
    
    # Analyze when experts disagree vs when they're correct
    ensemble_predictions = torch.mean(all_expert_logits, dim=1)  # [num_samples, num_classes]
    ensemble_pred_classes = torch.argmax(ensemble_predictions, dim=1)
    
    correct_ensemble = (ensemble_pred_classes == test_targets).sum().item()
    ensemble_accuracy = (correct_ensemble / num_samples) * 100
    
    print(f'Ensemble accuracy: {correct_ensemble}/{num_samples} ({ensemble_accuracy:.2f}%)')
    
    # Analyze cases where ensemble is correct vs individual experts
    individual_accuracies = []
    for expert_idx in range(num_experts):
        expert_pred = expert_predictions[:, expert_idx]
        correct = (expert_pred == test_targets).sum().item()
        accuracy = (correct / num_samples) * 100
        individual_accuracies.append(accuracy)
        print(f'Expert {expert_idx + 1} individual accuracy: {correct}/{num_samples} ({accuracy:.2f}%)')
    
    # Show improvement from ensemble
    best_individual = max(individual_accuracies)
    improvement = ensemble_accuracy - best_individual
    print(f'Ensemble improvement over best individual: {improvement:.2f}%')
    
    # Analyze diversity in wrong cases
    wrong_ensemble_indices = (ensemble_pred_classes != test_targets).nonzero(as_tuple=True)[0]
    if len(wrong_ensemble_indices) > 0:
        print(f'\nAnalyzing {len(wrong_ensemble_indices)} cases where ensemble was wrong:')
        
        expert_agreement_in_wrong_cases = 0
        for idx in wrong_ensemble_indices:
            expert_preds_for_sample = expert_predictions[idx]
            if torch.all(expert_preds_for_sample == expert_preds_for_sample[0]):
                expert_agreement_in_wrong_cases += 1
        
        wrong_agreement_rate = (expert_agreement_in_wrong_cases / len(wrong_ensemble_indices)) * 100
        print(f'Expert agreement rate in wrong cases: {expert_agreement_in_wrong_cases}/{len(wrong_ensemble_indices)} ({wrong_agreement_rate:.2f}%)')
        
        if wrong_agreement_rate > 50:
            print('High agreement in wrong cases suggests need for more diversity!')
        else:
            print('Good diversity even in wrong cases - experts are learning different perspectives.')


def main():
    """
    Main inference function
    """
    global train_set_file, test_set_file, path_rst
    
    parser = argparse.ArgumentParser(description="Enhanced CCNet Inference")
    
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--id_num", type=int, default=600, 
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500")
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--weight_chan", type=float, default=0.8, help="The weight of channel competition branch")
    
    # File paths
    parser.add_argument("--train_set_file", type=str, default='./data/train_all_server.txt')
    parser.add_argument("--test_set_file", type=str, default='./data/test_server.txt')
    parser.add_argument("--des_path", type=str, default='/data/YZY/Palm_DOC/Tongji_add/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='/data/YZY/Palm_DOC/Tongji_add/rst_test/')
    parser.add_argument("--check_point", type=str, default='/data/YZY/Palm_Doc/enhanced_net_params_best.pth')
    
    args = parser.parse_args()
    
    # Set global variables
    train_set_file = args.train_set_file
    test_set_file = args.test_set_file
    path_rst = args.path_rst
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    batch_size = args.batch_size
    num_classes = args.id_num
    
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)
    
    print('Enhanced CCNet Inference Configuration:')
    print(f'  Number of classes: {num_classes}')
    print(f'  Number of experts: {args.num_experts}')
    print(f'  Channel competition weight: {args.weight_chan}')
    print(f'  Model checkpoint: {args.check_point}')
    print(f'  Results path: {path_rst}')
    
    # Load datasets
    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)
    
    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=0, shuffle=True)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    # Load enhanced model
    print('Loading Enhanced CCNet model...')
    net = EnhancedCCNet(num_classes=num_classes, weight=args.weight_chan, num_experts=args.num_experts)
    
    # Load checkpoint
    if os.path.exists(args.check_point):
        print(f'Loading checkpoint from: {args.check_point}')
        net.load_state_dict(torch.load(args.check_point), strict=False)
        print('Checkpoint loaded successfully!')
    else:
        print(f'Warning: Checkpoint file not found at {args.check_point}')
        print('Using randomly initialized model...')
    
    # Run inference
    test_enhanced_model(net)
    
    print('Enhanced CCNet inference completed!')


if __name__ == "__main__":
    main()