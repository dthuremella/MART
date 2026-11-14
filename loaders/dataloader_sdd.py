import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

use_kalman = False
which_score = 'performance' 
if '6nn' not in which_score: which_score = which_score + '_redoint'

no_stationary = False

augment = True

def rot(t, theta, origin=np.array([1990.0/2, 1990.0/2])):
    t_trans = t - origin
    dx = t_trans[...,0]
    dy = t_trans[...,1]
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    rotated_t = np.stack((x_rot, y_rot), axis=-1) + origin
    return rotated_t

class TrajectoryDataset(Dataset):
    def __init__(
        self, obs_len=8, pred_len=12, mode='train', scale=10, inputs=None
    ):
        super(TrajectoryDataset, self).__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.scale = scale
        
        with open('./datasets/stanford/sdd_{}.pkl'.format(mode), 'rb') as f:
            traj = pickle.load(f)
        
        traj_tmp = []
        
        for t in traj:
            traj_tmp.append(t)
            if mode == 'train':
                traj_tmp.append(np.flip(t, axis=1))
        
        self.traj = []
        if 'pos_x' in inputs and 'pos_y' in inputs:
            for t in traj_tmp:
                t -= t[:, :1, :]
                self.traj.append(t)
        else:
            self.traj = traj_tmp

        if no_stationary and mode =='train':
            non_stationary = [np.any(np.linalg.norm(
                                scenario[:,1:,:] - scenario[:,:-1,:], axis=-1) > 0) for scenario in self.traj]
            self.traj = [t for t, keep in zip(self.traj, non_stationary) if keep]

        if use_kalman:
            data_root_kalman = './datasets/stanford/sdd_{}_{}.npy'.format(mode, which_score)
            self.kalman_score = np.load(data_root_kalman, allow_pickle=True)

        if augment and mode == 'train':
            # data_root_kalman = './datasets/stanford/sdd_{}_{}.npy'.format(mode, which_score)
            # kalman_score = np.load(data_root_kalman, allow_pickle=True)
            traj_aug = []
            for t in self.traj:
                traj_aug.append(t)
                for i in range(2): # flip horiz and vert
                    s = t.copy()
                    s[:,:,i] = -s[:,:,i] + 1990 # max pixel value (in both x and y)
                    traj_aug.append(s)
                for i in range(1,4): # rotate by 90, 180, 270
                    s = t.copy()
                    theta = torch.pi / 2 * i
                    traj_aug.append(rot(s, theta))
            self.traj = traj_aug
        
        
    def __len__(self):
        return len(self.traj)

    def __getitem__(self, index):
        past_traj = self.traj[index][:, :self.obs_len] * self.scale
        future_traj = self.traj[index][:, self.obs_len:] * self.scale
        past_traj = torch.from_numpy(past_traj).type(torch.float)
        future_traj = torch.from_numpy(future_traj).type(torch.float)

        if use_kalman: # (use_kalman and self.mode in ['train', 'val']):
            kalman_difficulty = self.kalman_score[index]
            return [past_traj, future_traj, kalman_difficulty]
        
        return [past_traj, future_traj]