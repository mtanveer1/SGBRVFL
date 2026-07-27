import os
import argparse
from gb_division import gb_division
import torch
import numpy as np
from tools.split_indices import split_indices
from split_test import get_planetoid_dataset, get_ogb_dataset
from split_test import get_coauthor_dataset
import copy
from SGBRVFL import SGBRVFL
import warnings
warnings.filterwarnings("ignore")
from SGBRVFL_Without import SGBRVFL_Without

def main():
    #parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv")   # ACM, Actor, Chameleon, CoraFull, Cora, Squirrel, UAI
    parser.add_argument('--split_data', type=str, default="normal")    # ogbn-arxiv, ogbn-mag, ogbn-proteins, ogbn-products, ogbn-papers100M
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--models', type=str, default='GCN')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--ball_r', type=float, default=0.3)
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--heads', type=int, default=8)

    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #dataset selection
    if args.dataset == "physics" or args.dataset == "cs":
        dataset = get_coauthor_dataset(args.dataset, split=args.split_data)
    if args.dataset == "ogbn-arxiv" or args.dataset == "ogbn-products":
        dataset = get_ogb_dataset(args.dataset, split=args.split_data)
    else:
        dataset = get_planetoid_dataset(args.dataset, split=args.split_data)

    data = dataset
    fun_data = copy.deepcopy(data)

    #GB division
    new_data = gb_division(fun_data, args)

    data = data.to(device)
    args.num_classes = len(set(np.array(data.y.cpu())))
    args.gb_labels = new_data['gb_labels']
    features = torch.from_numpy(new_data['gb_features'])
    args.num_features = len(features[0])

    all_acc = []
    precision=[]
    recall=[]
    f1 =[]
    auc =[]

    mew = torch.logspace(-5, 5, steps=11, base=2)
    C1 = torch.logspace(-5, 5, steps=11, base=10)
    N = list(range(3, 204, 20))

    #training
    for i in range(args.runs):
        train_list, val_list = split_indices(list(range(len(new_data['gb_labels']))), 20, ways="random")
        train_index = torch.tensor(train_list).to(device)
        val_index = torch.tensor(val_list).to(device)
        new_data['train_mask'] = torch.zeros(len(new_data['gb_labels']), dtype=torch.bool)
        new_data['val_mask'] = torch.zeros(len(new_data['gb_labels']), dtype=torch.bool)
        new_data['train_mask'][train_index] = True
        new_data['val_mask'][val_index] = True
        new_features = features.to(torch.float)
        new_adj = torch.from_numpy(new_data['adj']).to(torch.int64)
        new_labels = torch.from_numpy(new_data['gb_labels']).to(torch.int64)
        new_features = new_features.to(device)
        new_adj = new_adj.to(device)
        new_labels = new_labels.to(device)
        new_data['train_mask'] = new_data['train_mask'].to(device)
        new_data['val_mask'] = new_data['val_mask'].to(device)
       
        max_acc = 0
        for c_1 in C1:
       
            REG_PARAM=1/c_1
            for sigma in mew:
                for N1 in N:
                    base_model = SGBRVFL( sigma, reg_para=REG_PARAM)
                    pred = base_model.fit(new_features, new_labels, new_adj, new_data['train_mask'], new_data['val_mask'], N1, device)
                    # pred = base_model.fit(data.x, data.y, data.edge_index, data.train_mask, data.test_mask, N1, device)
                    print(pred[0])

                    if pred[0] > max_acc:
                        max_acc = pred[0]
                        N1_best = N1
                        c_1_best = REG_PARAM
                        sigma_best = sigma
                        
        base_model1 = SGBRVFL_Without( sigma_best, reg_para=c_1_best)
        pred = base_model1.fit(new_features, new_labels, new_adj, data.x, data.edge_index, data.y, N1_best, device)
        print(pred[0])

        all_acc.append(pred[0])
        precision.append(pred[1])
        recall.append(pred[2])
        f1.append(pred[3])
        auc.append(pred[4])
    all_acc_tensor = torch.tensor(all_acc)
    precision = torch.tensor(precision)
    recall = torch.tensor(recall)
    f1 = torch.tensor(f1)
    
    print('ave_acc: {:.4f}'.format(torch.mean(all_acc_tensor).item()), '+/- {:.4f}'.format(torch.std(all_acc_tensor).item()))
    print('ave_precision: {:.4f}'.format(torch.mean(precision).item()), '+/- {:.4f}'.format(torch.std(precision).item()))
    print('ave_recall: {:.4f}'.format(torch.mean(recall).item()), '+/- {:.4f}'.format(torch.std(recall).item()))
    print('ave_f1: {:.4f}'.format(torch.mean(f1).item()), '+/- {:.4f}'.format(torch.std(f1).item()))
 

if __name__ == '__main__':
    main()

