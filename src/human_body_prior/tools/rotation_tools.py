import numpy as np
import torchgeometry as tgm
from torch.nn import functional as F

def em2euler(em):
    '''

    :param em: rotation in expo-map (3,)
    :return: rotation in euler angles (3,)
    '''
    from transforms3d.euler import axangle2euler

    theta = np.sqrt((em ** 2).sum())
    axis = em / theta
    return np.array(axangle2euler(axis, theta))


def euler2em(ea):
    '''

    :param ea: rotation in euler angles (3,)
    :return: rotation in expo-map (3,)
    '''
    from transforms3d.euler import euler2axangle
    axis, theta = euler2axangle(*ea)
    return np.array(axis*theta)


def remove_zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: Nx3
    '''
    bs = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot, [0,1])
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot)
    return pose

def aa2matrot(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    bs = pose.size(0)
    num_joints = pose.size(1)//3
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()#.view(bs, num_joints*9)
    return pose_body_matrot

def noisy_zrot(rot_in):
    '''

    :param rot_in: np.array Nx3 rotations in axis-angle representation
    :return:
        will add a degree from a full circle to the zrotations
    '''
    is_batched = False
    if rot_in.ndim == 2: is_batched = True
    if not is_batched:
        rot_in = rot_in[np.newaxis]

    rnd_zrot = np.random.uniform(-np.pi, np.pi)
    rot_out = []
    for bId in range(len(rot_in)):
        pose_cpu = rot_in[bId]
        pose_euler = em2euler(pose_cpu)

        pose_euler[2] += rnd_zrot

        pose_aa = euler2em(pose_euler)
        rot_out.append(pose_aa.copy())

    return np.array(rot_out)

def rotate_points_xyz(mesh_v, Rxyz):
    '''

    :param mesh_v: Nxnum_vx3
    :param Rxyz: Nx3
    :return:
    '''

    mesh_v_rotated = []

    for fId in range(mesh_v.shape[0]):
        angle = np.radians(Rxyz[fId, 0])
        rx = np.array([
            [1., 0., 0.           ],
            [0., np.cos(angle), -np.sin(angle)],
            [0., np.sin(angle), np.cos(angle) ]
        ])

        angle = np.radians(Rxyz[fId, 1])
        ry = np.array([
            [np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.           ],
            [-np.sin(angle), 0., np.cos(angle)]
        ])

        angle = np.radians(Rxyz[fId, 2])
        rz = np.array([
            [np.cos(angle), -np.sin(angle), 0. ],
            [np.sin(angle), np.cos(angle), 0. ],
            [0., 0., 1. ]
        ])
        mesh_v_rotated.append(rz.dot(ry.dot(rx.dot(mesh_v[fId].T))).T)

    return np.array(mesh_v_rotated)