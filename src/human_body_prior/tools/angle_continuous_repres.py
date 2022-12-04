import torch.nn.functional as F
import torch
from torch import nn

import numpy as np

# numpy implementation of yi zhou's method
def norm(v):
    return v/np.linalg.norm(v)

def gs(M):
    a1 = M[:,0]
    a2 = M[:,1]
    b1 = norm(a1)
    b2 = norm((a2-np.dot(b1,a2)*b1))
    b3 = np.cross(b1,b2)
    return np.vstack([b1,b2,b3]).T

# input sz bszx3x2
def bgs(d6s):

    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:,:,0], p=2, dim=1)
    a2 = d6s[:,:,1]
    c = torch.bmm(b1.view(bsz,1,-1),a2.view(bsz,-1,1)).view(bsz,1)*b1
    b2 = F.normalize(a2-c,p=2,dim=1)
    b3=torch.cross(b1,b2,dim=1)
    return torch.stack([b1,b2,b3],dim=1).permute(0,2,1)

# class geodesic_loss_R(nn.Module):
#     def __init__(self,reduction='mean'):
#         super(geodesic_loss_R, self).__init__()
#
#         self.reduction = reduction
#         self.eps = 1e-6
#
#     # batch geodesic loss for rotation matrices
#     def bgdR(self,Rgts,Rps):
#         Rds = torch.bmm(Rgts.permute(0,2,1),Rps)
#         Rt = torch.sum(Rds[:,torch.eye(3).byte()],1) #batch trace
#         # necessary or it might lead to nans and the likes
#         theta = torch.clamp(0.5*(Rt-1), -1+self.eps, 1-self.eps)
#         return torch.acos(theta)
#
#     def forward(self, ypred, ytrue):
#         theta = self.bgdR(ypred,ytrue)
#         if self.reduction == 'mean':
#             return torch.mean(theta)
#         else:
#             return theta

class geodesic_loss_R(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(geodesic_loss_R, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta

# if __name__ == '__main__':
#     from torch.nn import Parameter
#     from pyquaternion import Quaternion
#     import torch.optim as optim
#     from torch.autograd import Variable
#     from torch import FloatTensor
# 
# #     #######################
# #     np.random.seed(3434)
# #
# #     from human_body_prior.tools.rotation_tools import matrot2aa, aa2matrot
# #
# #     Rgts = aa2matrot(torch.Tensor(np.asarray([0,0,0]).reshape(1,3).astype(np.float32)))
# #     d6s = torch.tensor([[ 7.2144e-01,  5.2910e-01],
# #         [-4.1014e-05,  2.0887e-01],
# #         [ 4.1067e-04, -4.1738e-04]], dtype=torch.float32).view(1,3,2)
# #     print(d6s.shape)
# #     print(bgs(d6s))
# 
#     np.random.seed(3434)
# 
#     R1 = Quaternion.random().rotation_matrix
#     R2 = Quaternion.random().rotation_matrix
# 
#     Rgts = np.stack([R1,R2]).astype(np.float32)
#     d6s = np.random.uniform(-0.5,0.5,size=(2,3,2)).astype(np.float32)
# 
#     Rgts = torch.from_numpy(Rgts) # gt rotations
#     d6s = torch.from_numpy(d6s) # random 6d representation
# 
#     geodesic_loss = geodesic_loss_R()
#     P = Parameter(FloatTensor(d6s), requires_grad=True)
#     optimizer = optim.Adam([P], 0.01)
#     Rgts = Variable(Rgts)
# 
#     steps = 0
#     while True:
#         Rgs = bgs(P)
#         loss = geodesic_loss(Rgs, Rgts)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         steps += 1
#         print('%d %s' % (steps, loss.item()))
#         if loss.item() < 0.01:
#             print('done')
#             break