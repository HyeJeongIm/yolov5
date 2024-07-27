import numpy as np
import argparse 
import torch

from train_dsvdd import TrainerDeepSVDD
# from preprocess import get_mnist

# from load_data import load_data
from car import get_car
import visualization
import os

from test import eval
import random
import ipdb

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

if __name__ == '__main__':
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=150, help="number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6,help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3,help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=list, default=[50],help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True,help='Pretrain the network using an autoencoder')
    parser.add_argument('--latent_dim', type=int, default=128,help='Dimension of the latent variable z')
    parser.add_argument('--normal_class', type=int, default=0,help='Class to be treated as normal. The rest will be considered as anomalous.')
    parser.add_argument('--abnormal_class', type=int, default=1,help='Class to be treated as normal. The rest will be considered as anomalous.')

    # dataset
    parser.add_argument("--dataset", type=str, default="car", help="type of dataset")
    parser.add_argument("--num_of_classes", type=int, default=10, help="# of class")

    args = parser.parse_args() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## write result
    result_dir = f'result/{args.dataset}'
    result_file_path = os.path.join(result_dir, 'auc_scores.txt')    
    os.makedirs(result_dir, exist_ok=True)

    with open(result_file_path, 'w') as file:
        file.write(f'Normal Class / Abnormal Class : ROC AUC Score\n')

    '''
        Data
    '''
    data = get_car(args)

    deep_SVDD = TrainerDeepSVDD(args, data, device)

    if args.pretrain:
        deep_SVDD.pretrain()
    deep_SVDD.train()

    '''
        test and visualization
    '''
        
    state_dict = torch.load(deep_SVDD.trained_weights_path)
    deep_SVDD.net.load_state_dict(state_dict['net_dict'])
    deep_SVDD.c = torch.Tensor(state_dict['center']).to(deep_SVDD.device)

    indices, labels, scores = eval(deep_SVDD.net, deep_SVDD.c, data[1], device)

    # normal 및 abnormal 점수 분리
    normal_scores = [score for label, score in zip(labels, scores) if label == 0]
    abnormal_scores = [score for label, score in zip(labels, scores) if label == 1]

    # 점수 분포 시각화
    visualization.distribution_normal(normal_scores, result_dir)
    visualization.distribution_abnormal(abnormal_scores, result_dir)
    visualization.distribution_comparison(normal_scores, abnormal_scores, result_dir)


    # visualization.auroc(labels, scores)
    visualization.auroc_confusion_matrix(args, labels, scores, result_file_path)
    visualization.top5_down5_visualization(args, indices, labels, scores, data)



