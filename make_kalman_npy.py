import torch
import numpy as np


def kalman_filter(history, prediction_horizon):
    """
    Batched Kalman filter that exactly matches the original single-sample version,
    extended to predict multiple future steps.

    :param history: Tensor of shape (N, M, 2)
    :param prediction_horizon: int — number of future steps to predict (C)
    :return: Tensor of shape (N, C, 2)
    """
    assert history.ndim == 3 and history.shape[2] == 2, \
        "history must be of shape (N, M, 2)"

    N, M, _ = history.shape

    z_x = history[:, :, 0].clone()
    z_y = history[:, :, 1].clone()

    # --- Handle NaNs exactly as original ---
    nan_mask = torch.isnan(z_x) | torch.isnan(z_y)
    num_nans = nan_mask.sum(dim=1)  # (N,)

    # Replace NaNs with zeros to avoid NaN propagation
    z_x = torch.nan_to_num(z_x, nan=0.0)
    z_y = torch.nan_to_num(z_y, nan=0.0)

    # Compute velocities exactly like original
    v_x = torch.zeros(N, device=history.device)
    v_y = torch.zeros(N, device=history.device)

    for i in range(M - 1):
        valid = ~nan_mask[:, i]
        v_x += valid * (z_x[:, i + 1] - z_x[:, i])
        v_y += valid * (z_y[:, i + 1] - z_y[:, i])

    effective_length = (M - num_nans).clamp_min(2)  # avoid divide-by-zero
    v_x /= (effective_length - 1)
    v_y /= (effective_length - 1)

    # Truncate leading NaNs
    min_nan = num_nans.min().item()
    z_x = z_x[:, min_nan:]
    z_y = z_y[:, min_nan:]
    length_history = M - min_nan

    # --- Allocate state ---
    x_x = torch.zeros((N, length_history + 1), device=history.device)
    x_y = torch.zeros((N, length_history + 1), device=history.device)
    P_x = torch.zeros_like(x_x)
    P_y = torch.zeros_like(x_y)
    P_vx = torch.zeros_like(x_x)
    P_vy = torch.zeros_like(x_y)

    P_x[:, 0] = 1.0
    P_y[:, 0] = 1.0
    P_vx[:, 0] = 1.0
    P_vy[:, 0] = 1.0
    x_x[:, 0] = z_x[:, 0]
    x_y[:, 0] = z_y[:, 0]

    Q = 0.00001
    R = 0.0001

    # --- Run Kalman filter exactly like original ---
    for k in range(length_history - 1):
        x_x[:, k + 1] = x_x[:, k] + v_x
        x_y[:, k + 1] = x_y[:, k] + v_y
        P_x[:, k + 1] = P_x[:, k] + P_vx[:, k] + Q
        P_y[:, k + 1] = P_y[:, k] + P_vy[:, k] + Q
        P_vx[:, k + 1] = P_vx[:, k] + Q
        P_vy[:, k + 1] = P_vy[:, k] + Q

        K_x = P_x[:, k + 1] / (P_x[:, k + 1] + R)
        K_y = P_y[:, k + 1] / (P_y[:, k + 1] + R)

        x_x[:, k + 1] = x_x[:, k + 1] + K_x * (z_x[:, k + 1] - x_x[:, k + 1])
        x_y[:, k + 1] = x_y[:, k + 1] + K_y * (z_y[:, k + 1] - x_y[:, k + 1])

        P_x[:, k + 1] = P_x[:, k + 1] - K_x * P_x[:, k + 1]
        P_y[:, k + 1] = P_y[:, k + 1] - K_y * P_y[:, k + 1]

        K_vx = P_vx[:, k + 1] / (P_vx[:, k + 1] + R)
        K_vy = P_vy[:, k + 1] / (P_vy[:, k + 1] + R)
        P_vx[:, k + 1] = P_vx[:, k + 1] - K_vx * P_vx[:, k + 1]
        P_vy[:, k + 1] = P_vy[:, k + 1] - K_vy * P_vy[:, k + 1]

    # --- Multi-step future prediction (C steps ahead) ---
    k = length_history - 1
    steps = torch.arange(1, prediction_horizon + 1, device=history.device).float().view(1, -1)

    x_future = x_x[:, k].unsqueeze(1) + v_x.unsqueeze(1) * steps
    y_future = x_y[:, k].unsqueeze(1) + v_y.unsqueeze(1) * steps

    predictions = torch.stack([x_future, y_future], dim=-1)  # (N, C, 2)
    return predictions


def main(data_from='./datasets/nba/nba_test.npy', data_to='./datasets/nba/nba_test_kalman.npy', T_hist=10, T_fut=20, # set for NBA as default
                    ade=False, intervals=[0,1,2,5,10,20,100]):  
    trajs = np.load(data_from) 
    trajs /= (94/28) # Turn to meters
    traj_abs = torch.from_numpy(trajs).type(torch.float)
    traj_abs = traj_abs.permute(0, 2, 1, 3)  #[40000, 11, 30, 2]
    traj_serialized = traj_abs.flatten(0,1)
    hist = traj_serialized[:,:T_hist,:]
    kalman_preds = kalman_filter(hist, T_fut)
    fut = traj_serialized[:,T_hist:,:]

    # calculate ADE or FDE
    if ade:
        kalman_ade = torch.norm((fut - kalman_preds), dim=-1).mean(dim=-1)
        kalman_ade = kalman_ade.reshape(traj_abs[...,0,0].shape)
        arr = np.array(kalman_ade)
    else:
        kalman_fde = torch.norm((fut[:,-1,:] - kalman_preds[:,-1,:]), dim=-1)
        kalman_fde = kalman_fde.reshape(traj_abs[...,0,0].shape)
        arr = np.array(kalman_fde)
    np.save(data_to, arr)

    # return kalman intervals
    return np.percentile(arr, intervals)



if __name__ == "__main__":
    # mode = 'test' # test or train
    dataset = 'nba'
    ade = False

    intervals=[0, 1, 2, 5, 10, 20, 100]

    for mode in ['train', 'test']:
        data_from = f'./datasets/{dataset}/{dataset}_{mode}.npy'
        data_to = f'./datasets/{dataset}/{dataset}_{mode}_kalman.npy'
        kalman_intervals = main(data_from=data_from, data_to=data_to, T_hist=10, T_fut=20, ade=ade)
        print(f'Saved to {data_to}')
        print(f'Nth percentile: {intervals}')
        print([kalman_interval for kalman_interval in kalman_intervals])