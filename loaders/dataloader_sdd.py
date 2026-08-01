import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

use_kalman = False
which_score = 'performance' 

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

        if use_kalman:
            data_root_kalman = './datasets/stanford/sdd_{}_{}.npy'.format(mode, which_score)
            self.kalman_score = np.load(data_root_kalman, allow_pickle=True)
        
        
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