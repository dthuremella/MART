"""
USAGE 

python make_performance_npy.py 
"""

import os
import random
import argparse

from csv import writer

import torch
import numpy as np
import pickle

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import MART
from loaders.dataloader_nba import NBADataset, attribute_dataset, use_kalman
from fmoe.megatron import fmoefy

def register_flop_hooks(model):
    def count_expert_flops(module, input, output):
        inp = input[0]
        tokens = inp.shape[0]
        d_model = inp.shape[1]
        d_hidden = module.htoh4.out_feat
        flops = 2 * tokens * (d_model * d_hidden + d_hidden * d_model)
        module._flop_count += flops

    def count_linear_flops(module, input, output):
        inp = input[0]
        tokens = inp.numel() // inp.shape[-1]
        flops = 2 * tokens * inp.shape[-1] * output.shape[-1]
        module._flop_count += flops

    for name, module in model.named_modules():
        if type(module).__name__ in ('_Expert', '_ExpertPrint'):
            module._flop_count = 0
            module.register_forward_hook(count_expert_flops)
        elif type(module) == torch.nn.Linear:
            module._flop_count = 0
            module.register_forward_hook(count_linear_flops)

def print_flops(model):
    total_flops = sum(m._flop_count for m in model.modules() if hasattr(m, '_flop_count'))
    print(f"\n=== FLOPs Report ===")
    print(f"Total FLOPs: {total_flops:.3e}")
    print("\nPer module breakdown:")
    for name, module in model.named_modules():
        if hasattr(module, '_flop_count') and module._flop_count > 0:
            print(f"  {name}: {module._flop_count:.3e}")

def get_total_flops(model):
    return sum(m._flop_count for m in model.modules() if hasattr(m, '_flop_count'))

def reset_flop_counts(model):
    for m in model.modules():
        if hasattr(m, '_flop_count'):
            m._flop_count = 0

def main(mode='train'):
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)

    print('[INFO] The seed is:', seed)
        
    dataset_test = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode=mode, end_train=40000)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=8)

    model = MART(opts).cuda()
    # model = fmoefy(model, fmoe_num_experts=8)
    # print(model)
    print('[INFO] Model params: {}'.format(sum(p.numel() for p in model.parameters())))

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)

    if opts.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.decay_step, gamma=opts.decay_gamma)
    elif opts.scheduler_type == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=opts.decay_gamma)

    model_save_dir = os.path.join('./checkpoints', os.path.basename(args.config).split('.')[0] + args.tag) 
    os.makedirs(model_save_dir, exist_ok=True)

    model_name = args.dataset + '_ckpt_best.pth'
    model_path = os.path.join(model_save_dir, model_name)
    print('[INFO] Loading model from:', model_path)
    model_ckpt = torch.load(model_path)
    model.load_state_dict(model_ckpt['state_dict'], strict=True)
    ade, fde = test(model_ckpt['epoch'], model, loader_test)
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('./results', '{}_result.csv'.format(args.dataset)), 'w', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow([os.path.basename(args.config).split('.')[0], ade, fde])


def test(epoch, model, loader):
    model.eval()
    # for name, m in model.named_modules():
    #     if 'linear_net' in name:
    #         print(name, type(m).__name__)
    avg_meter = {'epoch': epoch, 'ade_1': 0, 'ade_2': 0, 'ade_3': 0, 'ade_4': 0, 'fde_1': 0, 'fde_2': 0, 'fde_3': 0, 'fde_4': 0, 'counter': 0}
    xs, ys, ypreds = [], [], []

    with torch.no_grad():
        batch_count = 0
        for _, data in enumerate(loader):
            batch_count += 1
            if use_kalman:
                x_abs, y, kalman = data
                x_abs, y, kalman = x_abs.cuda(), y.cuda(), kalman.cuda()      
            else:
                x_abs, y = data
                x_abs, y = x_abs.cuda(), y.cuda()       
            
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]

            y_pred, _, score = model(x_abs, x_rel)

            xs.append(x_abs)
            ys.append(y)
            ypreds.append(y_pred)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]
            
            ade_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :5] - y[:, :, :, :5], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 4:5] - y[:, :, :, 4:5], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            ade_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :10] - y[:, :, :, :10], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 9:10] - y[:, :, :, 9:10], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            ade_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :15] - y[:, :, :, :15], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 14:15] - y[:, :, :, 14:15], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            ade_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
                        
            avg_meter['ade_1'] += ade_1
            avg_meter['fde_1'] += fde_1
            avg_meter['ade_2'] += ade_2
            avg_meter['fde_2'] += fde_2
            avg_meter['ade_3'] += ade_3
            avg_meter['fde_3'] += fde_3
            avg_meter['ade_4'] += ade_4
            avg_meter['fde_4'] += fde_4
            
            avg_meter['counter'] += (num_agents * batch_size)

    xs, ys, ypreds = torch.cat(xs).cpu(), torch.cat(ys).cpu(), torch.cat(ypreds).cpu()
    fdes = torch.min(torch.linalg.norm((ys.unsqueeze(2).expand(ypreds.shape) - ypreds), dim=-1), dim=2)[0][:,:,-1].cpu()
    arr = np.array(fdes)
    data_to = f'./datasets/{args.dataset}/{args.dataset}_{loader.dataset.mode}_performance.npy'
    np.save(data_to, arr)
    intervals=[0,1,2,5,10,20,100]
    print(f'Nth percentile: {intervals}')
    performance_interval = np.percentile(arr, intervals)
    print([performance_interval for performance_interval in performance_interval])

    th = get_th(opts, model)
    print('\n[{}] Epoch {} th: {}'.format(loader.dataset.mode.upper(), epoch, th))
    print('[{}] minADE/minFDE (1.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_1'] / avg_meter['counter'], avg_meter['fde_1'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (2.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_2'] / avg_meter['counter'], avg_meter['fde_2'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (3.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_3'] / avg_meter['counter'], avg_meter['fde_3'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (4.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_4'] / avg_meter['counter'], avg_meter['fde_4'] / avg_meter['counter']))
    return avg_meter['fde_4'] / avg_meter['counter'], avg_meter['ade_4'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='nba', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_nba_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--tag', type=str, default="", help='log tag add-on to folder name')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for mode in ['train', 'test']:
        print(f'\n=== Evaluating {mode} set ===')
        main(mode=mode)
