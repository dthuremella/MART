import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from models.moe import kalman, kalman_score


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
        
        
    def __len__(self):
        return len(self.traj)

    def __getitem__(self, index):
        past_traj = self.traj[index][:, :self.obs_len] * self.scale
        future_traj = self.traj[index][:, self.obs_len:] * self.scale
        past_traj = torch.from_numpy(past_traj).type(torch.float)
        future_traj = torch.from_numpy(future_traj).type(torch.float)

        kalman_error = None
        if kalman:
            total_traj = self.traj[index] * self.scale
            total_traj = torch.from_numpy(total_traj).type(torch.float)
            kalman_error = kalman_score(total_traj, self.obs_len, self.pred_len)
            
        return [past_traj, future_traj, kalman_error]
