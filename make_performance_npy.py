"""
USAGE 

python make_performance_npy.py 
"""

import os
import argparse

from csv import writer

import torch
import numpy as np
import pickle

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import MART
from loaders.dataloader_nba import NBADataset
from loaders.dataloader_sdd import TrajectoryDataset as SDDTrajectoryDataset
from loaders.dataloader_eth import TrajectoryDataset as ETHTrajectoryDataset
from main_sdd import my_collate
from fmoe.megatron import fmoefy

def main_nba(mode='train'):
    seed = args.seed
    setup_seed(seed)

    print('[INFO] The seed is:', seed)
        
    dataset_test = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode=mode, end_train=40000)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=8)

    model = MART(opts).cuda()
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
    ade, fde = test_nba(model_ckpt['epoch'], model, loader_test)

def test_nba(epoch, model, loader):
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

    data_to = f'./datasets/{args.dataset}/{args.dataset}_{loader.dataset.mode}_{args.epithet}performance.npy'
    np.save(data_to, arr)
    intervals=[0,1,2,5,10,20,100]
    print(f'Nth percentile: {intervals}')
    performance_interval = np.percentile(arr, intervals)
    print([performance_interval for performance_interval in performance_interval])

    if learnable_targets:
        target_logits_final = {}
        encoders = (model.pair_encoders + model.hyper_encoders)
        encoder_types = (['pair'] * len(model.pair_encoders) + ['group'] * len(model.hyper_encoders))
        layer_numbers = list(range(len(model.pair_encoders))) * 2
        encoder_strings = [f'{type_i}{layer_i}' for type_i, layer_i in zip(encoder_types, layer_numbers)]
        for encoder, encoder_str in zip(encoders, encoder_strings):  # however you access your gates
            target_logits_final[encoder_str] = {}
            for layer in encoder.layers:
                if hasattr(layer, 'linear_net_n'):
                    target_logits_final[encoder_str]['linear_net_n'] = torch.round(F.softmax(layer.linear_net_n.gate.target_logits)*100).int().tolist()
                if hasattr(layer, 'linear_net2_e'):
                    target_logits_final[encoder_str]['linear_net2_e'] = torch.round(F.softmax(layer.linear_net2_e.gate.target_logits)*100).int().tolist()
        for key0 in target_logits_final:
            print(key0)
            for key1 in target_logits_final[key0]:
                print(f'{key1}: {target_logits_final[key0][key1]}')

    th = get_th(opts, model)
    print('\n[{}] Epoch {} th: {}'.format(loader.dataset.mode.upper(), epoch, th))
    print('[{}] minADE/minFDE (1.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_1'] / avg_meter['counter'], avg_meter['fde_1'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (2.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_2'] / avg_meter['counter'], avg_meter['fde_2'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (3.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_3'] / avg_meter['counter'], avg_meter['fde_3'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (4.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_4'] / avg_meter['counter'], avg_meter['fde_4'] / avg_meter['counter']))
    return avg_meter['fde_4'] / avg_meter['counter'], avg_meter['ade_4'] / avg_meter['counter']

def main_sdd(mode='train'):
    seed = args.seed
    setup_seed(seed)

    print('[INFO] The seed is:', seed)

    dataset_test = SDDTrajectoryDataset(mode=mode, scale=opts.scale, inputs=opts.inputs)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, collate_fn=my_collate, shuffle=False, num_workers=8)

    if 'reported' in args.tag:
        opts.inputs = ['vel_x', 'vel_y']

    model = MART(opts).cuda()
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
    ade, fde = test_sdd(model_ckpt['epoch'], model, loader_test, mode)

def test_sdd(epoch, model, loader, mode):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade': 0, 'fde': 0, 'counter': 0}
    xs, ys, ypreds, batch_idxs = [], [], [], []

    with torch.no_grad():
        batch_count = 0
        for i, data in enumerate(loader):
            batch_count += 1
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()           
            
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]
            
            dims = 2 if 'reported' in args.tag else 3
            y_pred, _, score = model(x_abs[...,:dims], x_rel[...,:dims])

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1], :2].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            x_flat = x_abs.flatten(0,1)
            mask = x_flat[:,-1,-1].bool() # most current timestep needs to be valid to be counted
            xs.append(x_flat[mask])
            ys.append(y.flatten(0,1).clone()[mask])
            ypreds.append(y_pred.flatten(0,1).clone()[mask])
            batch_idx_flat = torch.arange(x_flat.shape[0], device=x_flat.device) // num_agents  
            batch_idxs.append(batch_idx_flat[mask] + i * opts.batch_size)

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]
            
            mask = np.array(x_abs[:,:,-1,-1].cpu())
            ade = np.sum(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2) * mask)
            fde = np.sum(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2) * mask)
                        
            avg_meter['ade'] += ade
            avg_meter['fde'] += fde
            
            avg_meter['counter'] += mask.sum()
    
    ###### Calculate performance intervals and save ###########
    xs, ys, ypreds, batch_idxs = torch.cat(xs).cpu(), torch.cat(ys).cpu(), torch.cat(ypreds).cpu(),  torch.cat(batch_idxs).cpu()
    fdes = torch.min(torch.linalg.norm((ys.unsqueeze(1).expand(ypreds.shape) - ypreds), dim=-1), dim=1)[0][:,-1].cpu()
    fdes /= opts.scale
    scores = np.array(fdes)
    intervals=[0,1,2,5,10,20,100]
    print(f'Nth percentile: {intervals}')
    performance_interval = np.percentile(scores, intervals)
    print([performance_interval for performance_interval in performance_interval])

    data_to = f'./datasets/stanford/{args.dataset}_{mode}_{args.epithet}performance.npy'
    scores_list = [scores[batch_idxs == b] for b in range(batch_idxs[-1] + 1)]

    np.save(data_to, np.array(scores_list, dtype=object), allow_pickle=True)
    
    if learnable_targets:
        target_logits_final = {}
        encoders = (model.pair_encoders + model.hyper_encoders)
        encoder_types = (['pair'] * len(model.pair_encoders) + ['group'] * len(model.hyper_encoders))
        layer_numbers = list(range(len(model.pair_encoders))) * 2
        encoder_strings = [f'{type_i}{layer_i}' for type_i, layer_i in zip(encoder_types, layer_numbers)]
        for encoder, encoder_str in zip(encoders, encoder_strings):  # however you access your gates
            target_logits_final[encoder_str] = {}
            for layer in encoder.layers:
                if hasattr(layer, 'linear_net_n'):
                    target_logits_final[encoder_str]['linear_net_n'] = torch.round(F.softmax(layer.linear_net_n.gate.target_logits)*100).int().tolist()
                if hasattr(layer, 'linear_net2_e'):
                    target_logits_final[encoder_str]['linear_net2_e'] = torch.round(F.softmax(layer.linear_net2_e.gate.target_logits)*100).int().tolist()
        for key0 in target_logits_final:
            print(key0)
            for key1 in target_logits_final[key0]:
                print(f'{key1}: {target_logits_final[key0][key1]}')

    avg_meter['ade'] /= opts.scale
    avg_meter['fde'] /= opts.scale
    
    th = get_th(opts, model)
    print('\n[{}][{}] Epoch {} th: {}'.format(args.dataset.upper(), mode, epoch, th))
    print('[{}][{}] minADE/minFDE: {:.2f}/{:.2f}'.format(args.dataset.upper(), mode, avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']

def main_eth(mode='train'):
    seed = args.seed
    setup_seed(seed)

    print('[INFO] The seed is:', seed)

    data_root = os.path.join('./datasets/ethucy', args.dataset)

    dataset_test = ETHTrajectoryDataset(args, os.path.join(data_root, mode), obs_len=opts.past_length, pred_len=opts.future_length, skip=1)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, collate_fn=my_collate, shuffle=False, num_workers=8)

    if 'reported' in args.tag:
        opts.inputs = ['vel_x', 'vel_y']

    model = MART(opts).cuda()
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
    ade, fde = test_eth(model_ckpt['epoch'], model, loader_test, mode)

def test_eth(epoch, model, loader, mode):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade': 0, 'fde': 0, 'counter': 0}
    xs, ys, ypreds, batch_idxs = [], [], [], []

    with torch.no_grad():
        batch_count = 0
        for i, data in enumerate(loader):
            batch_count += 1
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()         
                
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]

            dims = 2 if 'reported' in args.tag else 3
            y_pred, _, score = model(x_abs[...,:dims], x_rel[...,:dims])

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1], :2].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            x_flat = x_abs.flatten(0,1)
            mask = x_flat[:,-1,-1].bool() # most current timestep needs to be valid to be counted
            xs.append(x_flat[mask])
            ys.append(y.flatten(0,1).clone()[mask])
            ypreds.append(y_pred.flatten(0,1).clone()[mask])
            batch_idx_flat = torch.arange(x_flat.shape[0], device=x_flat.device) // num_agents  
            batch_idxs.append(batch_idx_flat[mask] + i * opts.batch_size)

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]

            mask = np.array(x_abs[:,:,-1,-1].cpu())
            ade = np.sum(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2) * mask)
            fde = np.sum(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2) * mask)
                        
            avg_meter['ade'] += ade
            avg_meter['fde'] += fde
            
            avg_meter['counter'] += mask.sum()

    ###### Calculate performance intervals and save ###########
    xs, ys, ypreds, batch_idxs = torch.cat(xs).cpu(), torch.cat(ys).cpu(), torch.cat(ypreds).cpu(),  torch.cat(batch_idxs).cpu()
    fdes = torch.min(torch.linalg.norm((ys.unsqueeze(1).expand(ypreds.shape) - ypreds), dim=-1), dim=1)[0][:,-1].cpu()
    scores = np.array(fdes)
    intervals=[0,1,2,5,10,20,100]
    print(f'Nth percentile: {intervals}')
    performance_interval = np.percentile(scores, intervals)
    print([performance_interval for performance_interval in performance_interval])

    data_to = f'./datasets/ethucy/{args.dataset}_{mode}_{args.epithet}performance.npy'
    scores_list = [scores[batch_idxs == b] for b in range(batch_idxs[-1] + 1)]

    np.save(data_to, np.array(scores_list, dtype=object), allow_pickle=True)

    if learnable_targets:
        target_logits_final = {}
        encoders = (model.pair_encoders + model.hyper_encoders)
        encoder_types = (['pair'] * len(model.pair_encoders) + ['group'] * len(model.hyper_encoders))
        layer_numbers = list(range(len(model.pair_encoders))) * 2
        encoder_strings = [f'{type_i}{layer_i}' for type_i, layer_i in zip(encoder_types, layer_numbers)]
        for encoder, encoder_str in zip(encoders, encoder_strings):  # however you access your gates
            target_logits_final[encoder_str] = {}
            for layer in encoder.layers:
                if hasattr(layer, 'linear_net_n'):
                    target_logits_final[encoder_str]['linear_net_n'] = torch.round(F.softmax(layer.linear_net_n.gate.target_logits)*100).int().tolist()
                if hasattr(layer, 'linear_net2_e'):
                    target_logits_final[encoder_str]['linear_net2_e'] = torch.round(F.softmax(layer.linear_net2_e.gate.target_logits)*100).int().tolist()
        for key0 in target_logits_final:
            print(key0)
            for key1 in target_logits_final[key0]:
                print(f'{key1}: {target_logits_final[key0][key1]}')

    th = get_th(opts, model)
    print('\n[{}][{}] Epoch {} th: {}'.format(args.dataset.upper(), 'TEST', epoch, th))
    print('[{}][{}] minADE/minFDE: {:.2f}/{:.2f}'.format(args.dataset.upper(), 'TEST', avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_nba_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--tag', type=str, default="", help='log tag add-on to folder name')
    parser.add_argument('--epithet', type=str, default="", help='shortened version of tag')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for mode in ['test', 'train']:
        print(f'\n=== Evaluating {mode} set ===')
        if 'nba' in args.config:
            args.dataset = 'nba'
            main_nba(mode=mode)
        elif 'sdd' in args.config:
            args.dataset = 'sdd'
            main_sdd(mode=mode)
        elif 'eth' in args.config:
            for eth_dataset in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                args.dataset = eth_dataset
                main_eth(mode=mode)
