import torch
import torch.nn as nn
import numpy as np


def _to_one_hot(y, n_dims=None):
    """ 
    Take integer y (tensor or variable) with n dims and 
    convert it to 1-hot representation with n+1 dims
    """
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.size()[0], -1)
    
    return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot


class LSEP(torch.autograd.Function): 
    """
    Autograd function of LSEP loss. Appropirate for multi-label
    - Reference: Li+2017
      https://arxiv.org/pdf/1704.03135.pdf
    """
    
    @staticmethod
    def forward(ctx, input, target, max_num_trials = None):
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ## rank weight 
        rank_weights = [1.0/1]
        for i in range(1, label_size):
            rank_weights.append(rank_weights[i-1] + (1.0/i+1))

        if max_num_trials is None: 
            max_num_trials = target.size()[1] - 1

        ##
        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        
        ## summing over all negatives and positives
        loss = 0.
        for i in range(input.size()[0]): # loop over examples
            pos = np.array([j for j,pos in enumerate(positive_indices[i]) if pos != 0])
            neg = np.array([j for j,neg in enumerate(negative_indices[i]) if neg != 0])
            
            for j,pj in enumerate(pos):
                for k,nj in enumerate(neg):
                    loss += np.exp(input[i,nj]-input[i,pj])
        
        loss = torch.from_numpy(np.array([np.log(1 + loss)])).float()
        
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        
        return loss

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        loss = torch.autograd.Variable(ctx.loss, requires_grad = False)
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices

        fac  = -1 / loss
        grad_input = torch.zeros(input.size())
        
        ## make one-hot vectors
        one_hot_pos, one_hot_neg = [],[]
        
        for i in range(grad_input.size()[0]): # loop over examples
            pos_ind = np.array([j for j,pos in enumerate(positive_indices[i]) if pos != 0])
            neg_ind = np.array([j for j,neg in enumerate(negative_indices[i]) if neg != 0])
            
            one_hot_pos.append(_to_one_hot(torch.from_numpy(pos_ind),input.size()[1]))
            one_hot_neg.append(_to_one_hot(torch.from_numpy(neg_ind),input.size()[1]))
            
        ## grad
        for i in range(grad_input.size()[0]):
            for dum_j,phot in enumerate(one_hot_pos[i]):
                for dum_k,nhot in enumerate(one_hot_neg[i]):
                    grad_input[i] += (phot-nhot)*torch.exp(-input[i].data*(phot-nhot))
        ## 
        grad_input = torch.autograd.Variable(grad_input) * (grad_output * fac)

        return grad_input, None, None
    
#--- main class
class LSEPLoss(nn.Module): 
    def __init__(self): 
        super(LSEPLoss, self).__init__()
        
    def forward(self, input, target): 
        return LSEP.apply(input.cpu(), target.cpu())