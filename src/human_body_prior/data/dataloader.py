# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.01.02

import glob, os

import torch
from torch.utils.data import Dataset
from configer import Configer

class VPoserDS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, data_fields=[]):
        assert os.path.exists(dataset_dir)
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            if len(data_fields) != 0 and k not in data_fields: continue
            self.ds[k] = torch.load(data_fname).type(torch.float32)

        dataset_ps_fname = glob.glob(os.path.join(dataset_dir, '..', '*.ini'))
        if len(dataset_ps_fname):
            self.ps = Configer(default_ps_fname=dataset_ps_fname[0], dataset_dir=dataset_dir)

    def __len__(self):
        k = list(self.ds.keys())[0]
        return len(self.ds[k])

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        return data


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from basis_point_sets.bps import reconstruct_from_bps
    from psbody.mesh import MeshViewer, Mesh
    from grab.tools.vis_tools import colors, points_to_spheres
    from human_body_prior.body_model.body_model import BodyModel
    from smplx.lbs import batch_rodrigues
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    # from kornia.geometry.conversions import rotation_matrix_to_angle_axis as matrot2aa#Buggy
    # from kornia.geometry.conversions import angle_axis_to_rotation_matrix as aa2matrot#Buggy

    from human_body_prior.tools.rotation_tools import matrot2aa
    from human_body_prior.tools.rotation_tools import aa2matrot

    import time
    import numpy as np

    batch_size = 1
    dataset_dir = '/ps/project/human_body_prior/VPoser/data/007_00_00/smpl/pytorch/stage_III/test/'
    # dataset_dir = '/ps/scratch/body_hand_object_contact/grab_net/data/V01_07_00/train'

    ds = VPoserDS(dataset_dir=dataset_dir)
    print('dataset size: %d'%len(ds))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)

    mv = MeshViewer(keepalive=False)
    bm_path = '/ps/project/common/moshpp/smplx/unlocked_head/neutral/model.npz'

    bm = BodyModel(bm_path, batch_size=batch_size, num_betas=16)

    n_joints = 22

    id = 0

    def batched_aa2matrot(aas):
        matrots = []
        for bId in range(aas.shape[0]):
            matrots.append(aa2matrot(aas[bId]))
        return torch.stack(matrots)

    def batched_matrot2aa(matrots):
        aas = []
        for bId in range(matrots.shape[0]):
            aas.append(matrot2aa(matrots[bId]))
        return torch.stack(aas)

    for i_batch, dorig in enumerate(dataloader):
        # fullpose = dorig['fullpose'].view(batch_size,n_joints,3).view(-1,3)
        # fullpose_rec = matrot2aa(aa2matrot(fullpose)).contiguous().view(batch_size, n_joints, 3).view(batch_size,-1)
        # fullpose_rec = batched_matrot2aa(batched_aa2matrot(dorig['fullpose'].view(batch_size, -1,3))).view(batch_size,-1)#.contiguous().view(batch_size, n_joints, 3).view(batch_size,-1)
        # dorig_cpu = {k:c2c(v) for k, v in dorig.items()}
        
        # np.savez('/home/nghorbani/Downloads/fullpose.npz', fullpose=dorig_cpu['fullpose'])
        fullpose = torch.from_numpy(np.load('/home/nghorbani/Downloads/fullpose.npz')['fullpose'])
        fullpose_rec = matrot2aa(aa2matrot(fullpose.view(-1,3))).reshape(batch_size,-1)

        # bm_eval = bm(root_orient = dorig['fullpose'][:,:3], pose_body = dorig['fullpose'][:,3:66],).v
        bm_eval = bm(root_orient = fullpose[:,:3], pose_body = fullpose[:,3:66],).v
        bm_mesh = Mesh(c2c(bm_eval[id]), c2c(bm.f), vc=colors['grey'])


        bm_eval_rec = bm(root_orient=fullpose_rec[:, :3], pose_body=fullpose_rec[:, 3:66], ).v
        bm_mesh_rec = Mesh(c2c(bm_eval_rec[id]), c2c(bm.f), vc=colors['blue'])

        mv.set_static_meshes([bm_mesh, bm_mesh_rec])

        time.sleep(2)

