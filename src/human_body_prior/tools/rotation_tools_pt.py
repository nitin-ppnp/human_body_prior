import torch
import numpy as np

def local2global_pose(local_pose, kintree):
    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose

# if __name__ == '__main__':
#     kintree = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
# 
#     data = np.load('/is/ps2/nghorbani/code-repos/human_body_prior/tests/samples/body_pose_rnd.npz')['data']
#     bs = 1
#     fullpose = np.concatenate([np.zeros((bs,1,3)), data.reshape((bs,-1,3)), np.zeros((bs,2,3))], axis=1)
#     fullpose = torch.from_numpy(fullpose).type(torch.float32)
# 
#     from human_body_prior.tools.rotation_tools import aa2matrot
#     from human_body_prior.tools.omni_tools import copy2cpu as c2c
#     fullpose_matrot = aa2matrot(fullpose.view(-1,3)).view(bs, -1, 3, 3)
#     print(fullpose_matrot.shape)
#     a = local2global_pose(fullpose_matrot)
#     b = local2global_pose2(fullpose_matrot)
#     print(c2c(a==b).sum() == np.prod(fullpose_matrot.shape))

