import sys
import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def Evaluate(ACTUAL, PREDICTED):
    y_true = ACTUAL.cpu().detach().numpy().reshape(-1)
    y_pred = PREDICTED.reshape(-1)


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro') 
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    try:
        auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    except ValueError:
        auc = None  # AUC is not valid for single-class predictions

    
    EVAL = [accuracy, precision, recall, f1, auc]
    return EVAL

import torch

def create_sparse_adj(A_norm, y_tr):
    """
    Converts a sparse adjacency matrix (COO format) from SciPy to PyTorch.

    Args:
        A_norm (torch.Tensor): A 2D tensor containing edge indices (shape: [2, num_edges]).
        y_tr (torch.Tensor): A 1D tensor representing node labels (for shape reference).

    Returns:
        torch.sparse.Tensor: The adjacency matrix in sparse format.
    """
    num_nodes = y_tr.shape[0]  # Get number of nodes

    # Create edge indices
    row = A_norm[0]  # First row = source nodes
    col = A_norm[1]  # Second row = target nodes

    # Create values (all ones, like in SciPy)
    values = torch.ones(A_norm.shape[1], dtype=torch.float32, device=A_norm.device)

    # Create the sparse adjacency matrix
    A_torch = torch.sparse_coo_tensor(
        indices=torch.stack((row, col)),  # Stack row and col indices
        values=values,  # Values (all ones)
        size=(num_nodes, num_nodes),  # Shape of adjacency matrix
        dtype=torch.float32  # Data type
    ).coalesce()  # Optimize storage

    return A_torch


class SGBRVFL_Without:
    def __init__(self, sigma=1, reg_para=1.):
        self.reg_para = reg_para
        self.sigma = sigma
        self.H_tr=[]
        self.H_ts=[]
        self.B=[]
        self.y_pre=[]

    def rbf_kernel_X(self, X, gamma):
        n = X.shape[0]
        Sij = torch.matmul(X, X.T)
        Si = torch.unsqueeze(torch.diag(Sij), 0).T @ torch.ones(1, n).to(X.device)
        Sj = torch.ones(n, 1).to(X.device) @ torch.unsqueeze(torch.diag(Sij), 0)
        D2 = Si + Sj - 2 * Sij
        K = torch.exp(-D2 * gamma)
        # K[torch.isinf(K)] = 1.
        return K

    def rbf_kernel_K(self, K_t, gamma):
        n = K_t.shape[0]
        s = torch.unsqueeze(torch.diag(K_t), 0)
        D2 = torch.ones(n, 1).to(K_t.device) @ s + s.T @ torch.ones(1, n).to(K_t.device) - 2 * K_t
        K = torch.exp(-D2 * gamma)
        # K[torch.isinf(K)] = 1.
        return K
    
    def RFF(in_dim, out_dim):
        weights = nn.Parameter(torch.randn([in_dim, round(out_dim/2)]), requires_grad=False)
        x = x.matmul(weights)
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        D = out_dim
        z = z / math.sqrt(D)
        return z


    def fit(self, X, y_tr, A_norm, test, test_edge, y_test,N1, device):
        test_edge = create_sparse_adj(test_edge, y_test)

        num_classes = len(torch.unique(y_tr))
        y_tr = F.one_hot(y_tr, num_classes=num_classes)

        # num_nodes = X.shape[0]
        # values = torch.ones(A_norm.shape[1]) 
        # adjacency_matrix = torch.sparse_coo_tensor(A_norm, values, (num_nodes, num_nodes))
        # A_norm = adjacency_matrix.to_dense()

        # A_norm = sp.coo_matrix((np.ones(A_norm.shape[1]), (A_norm[0], A_norm[1])),
        #                              shape=(y_tr.shape[0], y_tr.shape[0]),
        #                              dtype=np.float32)

       
        A_norm = A_norm.to(torch.float32)  

        X_t = A_norm @ X
        Z1 = self.rbf_kernel_X(X_t, self.sigma)
                
        Hp = A_norm @ X
        [in_dim, out_dim] = Hp.shape

        weights = nn.Parameter(torch.randn([out_dim, N1], device=device), requires_grad=False)
        x = Hp.matmul(weights)
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        D = out_dim
        z2 = z / math.sqrt(D)
    
        H_ = torch.hstack((X_t, z2))

        X=H_
            
        H_tr = X

        y_tr = y_tr.to(device).float()

        if H_tr.shape[0] >= H_tr.shape[1]:
            TT = torch.matmul(H_tr.T, H_tr) + self.reg_para * torch.eye(H_tr.shape[0], device=H_tr.device)
            inv_ = torch.linalg.inv(TT)
            A = torch.matmul(torch.matmul(inv_, H_tr.T), y_tr)  # Ensure y_tr is Float

        else:
            TT = torch.matmul(H_tr, H_tr.T) + self.reg_para * torch.eye(H_tr.shape[0], device=H_tr.device)
            epsilon = 1e-2  # A small number
            identity_matrix = torch.eye(TT.size(0), device=device)
            TT_regularized = TT + epsilon * identity_matrix
            inv_ = torch.linalg.inv(TT_regularized)
            A = torch.matmul(torch.matmul(H_tr.T, inv_), y_tr.float())  # Ensure y_tr is Float
        
        ##################################################################
        test_edge = test_edge.to(torch.float32)  

        X_t1 = test_edge @ test
        Z11 = self.rbf_kernel_X(X_t1, self.sigma)
                
        Hpp = test_edge @ test
        [in_dim, out_dim] = Hpp.shape

    
        x = Hpp.matmul(weights)
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        z22 = z / math.sqrt(D)
    
        H_tr_i = torch.hstack((X_t1, z22))

        #######################################################

        y_pre1 = torch.matmul(H_tr_i, A)

        Validation_label = np.argmax(y_pre1.cpu().detach().numpy(), axis=1).reshape(-1, 1)
 
        EVAL_Validation = Evaluate(y_test.reshape(-1, 1), Validation_label)

        return EVAL_Validation
