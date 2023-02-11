import h5py
import numpy
import numpy as np
from kpts_mapping.gta import GTA_ORIGINAL_NAMES, GTA_KEYPOINTS
from kpts_mapping.gta_im import GTA_IM_NPZ_KEYPOINTS, GTA_IM_PKL_KEYPOINTS
from kpts_mapping.smplx import SMPLX_KEYPOINTS
from kpts_mapping.smpl import SMPL_KEYPOINTS
import pickle
import os
import cv2
from scipy.spatial.transform import Rotation as R

# step0:
# merge info_frames.pickle info_frames.npz
# trans world to camera


data_root = './'
rec_idx = '2020-06-11-10-06-48-add-transl'
info_pkl = pickle.load(open(os.path.join(data_root, rec_idx, 'info_frames.pickle'), 'rb'))
info_npz = np.load(open(os.path.join(data_root, rec_idx, 'info_frames.npz'), 'rb'))
kpts_npz = np.array(info_npz['joints_3d_world'])
kpts_pkl = np.array([i['kpvalue'] for i in info_pkl]).reshape(kpts_npz.shape[0], -1, kpts_npz.shape[2])
kpts_pkl_names = [i['kpname'] for i in info_pkl]
world2cam = np.array(info_npz['world2cam_trans'])
kpts_world = np.concatenate((kpts_npz, kpts_pkl), axis=1)
kpts_valid = np.zeros([len(kpts_world), len(kpts_world[0]), 1])
print(
    f'loading kpts...\nkpts_npz.shape, kpts_pkl.shape, kpts_world.shape: {kpts_npz.shape}, {kpts_pkl.shape}, {kpts_world.shape}')

# step0
# 3d kpts world system to camera system
kpts_camera = []
for i in range(len(kpts_world[0])):
    if sum(kpts_world[0, i, :]) != 0:
        kpts_valid[:, i] = 1
for i in range(len(kpts_world)):
    r_i = world2cam[i][:3, :3].T
    t_i = world2cam[i][3, :3]
    # cam_point_i = r_i * kpts_world[i] + t_i
    cam_point_i = [np.matmul(r_i, kpt) + t_i for kpt in kpts_world[i]]
    kpts_camera.append(cam_point_i)
kpts_camera = np.concatenate((kpts_camera, kpts_valid), axis=-1)

# step1
# gta_im kpts to gta kpts
kpts_gta_im = numpy.array(kpts_camera)
kpts_gta = numpy.zeros(shape=(len(kpts_gta_im), len(GTA_KEYPOINTS), 4))
gta_im_names = GTA_IM_NPZ_KEYPOINTS + GTA_IM_PKL_KEYPOINTS
mapping_list = []

for i in range(len(kpts_gta)):
    mapping_list_i = []
    gta_im_names = GTA_IM_NPZ_KEYPOINTS + kpts_pkl_names[i]
    for kpt_name in GTA_ORIGINAL_NAMES:
        if kpt_name not in gta_im_names:
            mapping_list_i.append(-1)
        else:
            mapping_list_i.append(gta_im_names.index(kpt_name))
    mapping_list.append(mapping_list_i)

for i in range(len(kpts_gta)):
    for j in range(len(kpts_gta[0])):
        if mapping_list[i][j] != -1:
            kpts_gta[i][j] = kpts_gta_im[i][mapping_list[i][j]]
# average for nose
for i in range(len(kpts_gta)):
    kpts_gta[i][-1] = np.average(kpts_gta[i][45:51], axis=0)

# step2
# gta kpts to smpl kpts
mapping_list2 = []
valid_len = 0
for kpt_name in SMPL_KEYPOINTS:
    if kpt_name not in GTA_KEYPOINTS:
        mapping_list2.append(-1)
    else:
        mapping_list2.append(GTA_KEYPOINTS.index(kpt_name))
        valid_len += 1
print(f'tansform to smpl joints, num kpts: {len(mapping_list2)}, valid kpts:{valid_len}')
# print(mapping_list2)
kpts_smpl = numpy.zeros(shape=(len(kpts_gta), len(SMPL_KEYPOINTS), 4))
for i in range(len(kpts_smpl)):
    for j in range(len(kpts_smpl[0])):
        if mapping_list2[j] != -1:
            kpts_smpl[i][j] = kpts_gta[i][mapping_list2[j]]

# step3: transfer kpts3d joints3d to 2d
smpl_pkl_folder = 'C:\\Users\\90532\\Desktop\\code\\smplx\\output'
kpts_3d = kpts_smpl
print('kpts3d.shape', kpts_3d.shape)
kpts_2d = []
joints_2d = []
joints_3d = []
rvec = np.zeros([3, 1], dtype=float)
tvec = np.zeros([3, 1], dtype=float)
dist_coeffs = np.zeros([4, 1], dtype=float)
for i in range(len(kpts_3d)):
    camera_matrix_i = info_npz['intrinsics'][i]
    kpts3d_i = np.array(kpts_3d[i, :, :3], dtype=float)

    filename = '{:03d}'.format(i + 1) + '.pkl'
    file_i = open(os.path.join(smpl_pkl_folder, filename), "rb")
    smpl_parm_i = pickle.load(file_i)
    joints3d_i = smpl_parm_i['joints'].cpu().detach().numpy().reshape(-1, 3)[:24]

    kpts2d_i, _ = cv2.projectPoints(kpts3d_i, rvec, tvec, camera_matrix_i, dist_coeffs)
    joints2d_i, _ = cv2.projectPoints(joints3d_i, rvec, tvec, camera_matrix_i, dist_coeffs)

    kpts_2d.append(kpts2d_i)
    joints_3d.append(joints3d_i)
    joints_2d.append(joints2d_i)

# add valid label
kpts_2d = np.squeeze(np.array(kpts_2d, dtype=float))
kpts_3d = np.array(kpts_3d, dtype=float)
valid_kpts = kpts_3d[:, :, -1].reshape(len(kpts_3d), -1, 1)
print('gt2d.shape, valid_kpts.shape', kpts_2d.shape, valid_kpts.shape)
kpts_2d = np.concatenate((kpts_2d, valid_kpts), axis=-1)

valid_joints = np.ones_like(valid_kpts)
joints_3d = np.array(joints_3d, dtype=float)
joints_2d = np.squeeze(np.array(joints_2d, dtype=float))
joints_2d = np.concatenate((joints_2d, valid_joints), axis=-1)
joints_3d = np.concatenate((joints_3d, valid_joints), axis=-1)




# step4: get smplx mesh pose and shape
smpl_poses_matrix = []
smpl_shapes = []
smpl_transl = []
for filename in os.listdir(smpl_pkl_folder):
    if filename.endswith('.pkl'):
        file_i = open(os.path.join(smpl_pkl_folder, filename), "rb")
        smpl_parm_i = pickle.load(file_i)
        pose_matrix_i = smpl_parm_i['full_pose'].cpu().detach().numpy().reshape(-1, 3, 3)
        shape_i = smpl_parm_i['betas'].cpu().detach().numpy().reshape(-1)
        transl_i = smpl_parm_i['transl'].cpu().detach().numpy().reshape(-1, 3)
        smpl_poses_matrix.append(pose_matrix_i)
        smpl_shapes.append(shape_i)
        smpl_transl.append(transl_i)
        file_i.close()

smpl_poses_matrix = np.array(smpl_poses_matrix).reshape(-1, 3, 3)
rot_matrix = R.from_matrix(smpl_poses_matrix)
smpl_poses_rotvec = rot_matrix.as_rotvec().reshape(-1, 24, 3)

smpl_shapes = np.array(smpl_shapes)
smpl_transl = np.array(smpl_transl)

# smpl_shapes = np.average(smpl_betas, axis=0)
# h5f_shapes = np.repeat(smpl_shapes, len(kpts_smplx), axis=0)

smpl_transl = smpl_transl.reshape(-1, 3)
smpl_shapes = smpl_shapes.reshape(-1, 10)
smpl_poses = smpl_poses_rotvec.reshape(-1, 72)
print('smpl_poses.shape, smpl_shapes.shape', smpl_poses.shape, smpl_shapes.shape)



hmr_anno_dict = {'shape': smpl_shapes, 'pose': smpl_poses, 'transl': smpl_transl, 'kpts_2d': kpts_2d, 'kpts_3d': kpts_3d, 'joints_2d':joints_2d, 'joints_3d':joints_3d}
import pickle

with open(os.path.join(data_root, rec_idx, 'info_hmr.pickle'), 'wb') as f:
    pickle.dump(hmr_anno_dict, f)


info_hmr = np.load(os.path.join(data_root, rec_idx, 'info_hmr.pickle'), allow_pickle=True)
print(info_hmr)

