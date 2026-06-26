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
from models.mart import MART, viz
from loaders.dataloader_nba import NBADataset, attribute_dataset, use_kalman
from fmoe.megatron import fmoefy
import time
from deepspeed.profiling.flops_profiler import FlopsProfiler
import torch.cuda.nvtx as nvtx

measure_nvtx = False 

measure_flops = False
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
    # print("\nPer module breakdown:")
    # for name, module in model.named_modules():
    #     if hasattr(module, '_flop_count') and module._flop_count > 0:
    #         print(f"  {name}: {module._flop_count:.3e}")

def get_total_flops(model):
    return sum(m._flop_count for m in model.modules() if hasattr(m, '_flop_count'))

def reset_flop_counts(model):
    for m in model.modules():
        if hasattr(m, '_flop_count'):
            m._flop_count = 0

def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)

    print('[INFO] The seed is:', seed)
    if not args.test:
        dataset_train = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode='train')
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=8, drop_last=True)
        
    dataset_test = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode='test' if args.test else 'val')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=8)

    model = MART(opts).cuda()
    # model = fmoefy(model, fmoe_num_experts=8)
    # print(model)
    print('[INFO] Model params: {}'.format(sum(p.numel() for p in model.parameters())))
    if measure_flops:
        register_flop_hooks(model)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)

    if opts.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.decay_step, gamma=opts.decay_gamma)
    elif opts.scheduler_type == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=opts.decay_gamma)

    model_save_dir = os.path.join('./checkpoints', os.path.basename(args.config).split('.')[0] + args.tag) 
    os.makedirs(model_save_dir, exist_ok=True)

    if args.test:
        model_name = args.dataset + '_ckpt_best.pth'
        model_path = os.path.join(model_save_dir, model_name)
        print('[INFO] Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)
        prof = FlopsProfiler(model) # flops
        ade, fde = test(model_ckpt['epoch'], model, loader_test, prof)
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('./results', '{}_result.csv'.format(args.dataset)), 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow([os.path.basename(args.config).split('.')[0], ade, fde])
        exit()

    results = {'epochs': [], 'losses': []}
    best_val_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    print('[INFO] The seed is :',seed)
    
    for epoch in range(0, opts.num_epochs):
        train(epoch, model, optimizer, loader_train)
        test_loss, ade = test(epoch, model, loader_test)
        results['epochs'].append(epoch)
        results['losses'].append(test_loss)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_ade = ade
            best_epoch = epoch
            file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_best.pth')
            torch.save(state, file_path)
        print('[INFO] Best {} Loss: {:.5f} \t Best ade: {:.5f} \t Best epoch {}\n'.format(loader_test.dataset.mode.capitalize(), best_val_loss, best_ade, best_epoch))

        file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch) + '.pth')
        if epoch > 0:
            remove_file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch - 1) + '.pth')
            os.system('rm ' + remove_file_path)
            
        torch.save(state, file_path)
        
        if opts.scheduler_type is not None:
            scheduler.step()


def train(epoch, model, optimizer, loader):
    model.train()
    avg_meter = {'epoch': epoch, 'loss': 0, 'counter': 0}
    loader_len = len(loader)

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        
        kalman_score = None
        if use_kalman:
            x_abs, y, kalman = data
            x_abs, y, kalman = x_abs.cuda(), y.cuda(), kalman.cuda()
            if force_kalman: 
                kalman_score = kalman   
        else:
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()        
        
        batch_size, num_agents, length, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]
        
        if (i % 100 == 0 and i != 0) and measure_flops: reset_flop_counts(model)
        y_pred, avg_expert_idx, _ = model(x_abs, x_rel, kalman_score=kalman_score)
        if (i % 100 == 0 and i != 0) and measure_flops: flops_per_batch = get_total_flops(model)

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos
            
        y = y[:, :, None, :, :]
        
        total_loss = torch.mean(torch.min(torch.mean(torch.norm(y_pred - y, dim=-1), dim=3), dim=2)[0]) # for all agents
        if harmonic_bias_loss is not None:
            total_loss += harmonic_bias_loss * avg_expert_idx
        if learnable_targets:
            learnable_target_alpha = 0.01  # weight for the learnable targets loss, original is 0.01
            lower_flops_alpha = 0.01  # weight for the lower flops loss, original is 0.01
            
            learnable_target_loss, lower_flops_loss = 0, 0
            # in your training_step, after computing the MoE loss:
            for encoder in (model.pair_encoders + model.hyper_encoders):  # however you access your gates
                for layer in encoder.layers:
                    layer_moes = []
                    if hasattr(layer, 'linear_net_n'): layer_moes.append(layer.linear_net_n)
                    if hasattr(layer, 'linear_net_2e'): layer_moes.append(layer.linear_net_2e)
                    for layer_moe in layer_moes:
                        if layer_moe.gate.target_logits is not None:
                            p = F.softmax(layer_moe.gate.target_logits, dim=0)
                            entropy = -(p * p.log()).sum()
                            learnable_target_loss = learnable_target_loss - learnable_target_alpha * entropy  # subtract to maximise entropy (prevent collapse)
                            if lower_flops:
                                target_flops = torch.sum(torch.tensor(ratios, device=p.device) * p)
                                lower_flops_loss = lower_flops_loss + lower_flops_alpha * target_flops

            total_loss = total_loss + learnable_target_loss + lower_flops_loss

        avg_meter['loss'] += total_loss.item() * batch_size * num_agents
        avg_meter['counter'] += (batch_size * num_agents)
        
        total_loss.backward()
        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
        optimizer.step()

        if i % 100 == 0:
            if i != 0 and measure_flops:
                print('[{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Flops: {} | LR: {}'
                    .format(loader.dataset.mode.upper(), epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), flops_per_batch, optimizer.param_groups[0]['lr']))
            else:
                th = get_th(opts, model)
                print('[{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Threshold: {} | LR: {}'
                    .format(loader.dataset.mode.upper(), epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), th, optimizer.param_groups[0]['lr']))
    return avg_meter['loss'] / avg_meter['counter']


def test(epoch, model, loader, prof=None):
    model.eval()
    # for name, m in model.named_modules():
    #     if 'linear_net' in name:
    #         print(name, type(m).__name__)
    avg_meter = {'epoch': epoch, 'ade_1': 0, 'ade_2': 0, 'ade_3': 0, 'ade_4': 0, 'fde_1': 0, 'fde_2': 0, 'fde_3': 0, 'fde_4': 0, 'counter': 0}
    if viz:
        scores = {}
        for score_type in ['gate_score', 'top_k_idx']:
            scores[score_type] = {'pair_n': [], 'pair_e': [], 'group_n': [], 'group_e': []}
        xs, ys, ypreds = [], [], []

    global measure_nvtx
    t0 = time.time()
    if prof: prof.start_profile()
    # if measure_nvtx: nvtx.range_push("forward")    # start measuring
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
            
            if batch_count == 2:
                if measure_nvtx:
                    torch.cuda.synchronize()
                    nvtx.range_push("forward")
                if measure_flops:
                    reset_flop_counts(model)

            y_pred, _, score = model(x_abs, x_rel)

            if batch_count == 2:
                if measure_nvtx:
                    torch.cuda.synchronize()
                    nvtx.range_pop()
                    break
                if measure_flops:
                    flops_per_batch = get_total_flops(model)
                    if args.test:
                        print_flops(model)
                        import sys; sys.exit(0)

            if viz:
                if moe_e or moe_n:
                    for score_type in ['gate_score', 'top_k_idx']:
                        for score_subtype in ['pair_n', 'pair_e', 'group_n', 'group_e']:
                            if score[score_type][score_subtype][0] is None: continue
                            scores[score_type][score_subtype].append(torch.stack(score[score_type][score_subtype]))
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
            # break #debugging
    # import sys; sys.exit(0) #debugging
    t1 = time.time(); print('INFO: Time taken for inference is :', t1 - t0)
    if prof: prof.stop_profile(); print('INFO: FLOPs for inference is :', prof.get_total_flops())
    if measure_nvtx: 
        nvtx.range_pop(); import sys; sys.exit(0) # stop measuring

    if viz and not measure_nvtx:
        if learnable_targets:
            for encoder in (model.pair_encoders + model.hyper_encoders):  # however you access your gates
                for layer in encoder.layers:
                    layer_moes = []
                    if hasattr(layer, 'linear_net_n'): layer_moes.append(layer.linear_net_n)
                    if hasattr(layer, 'linear_net_2e'): layer_moes.append(layer.linear_net_2e)
                    for layer_moe in layer_moes:
                        print('target_logits:', layer_moe.gate.target_logits)
                        # if layer_moe.gate.target_logits is not None:
                        #     p = F.softmax(layer_moe.gate.target_logits, dim=0)
                        #     entropy = -(p * p.log()).sum()
                    print('\n')
                print('hyper encoders:')

        if moe_e or moe_n:
            for score_type in ['gate_score', 'top_k_idx']:
                for score_subtype in ['pair_n', 'pair_e', 'group_n', 'group_e']:
                    if len(scores[score_type][score_subtype]) == 0: continue
                    scores[score_type][score_subtype] = torch.cat(scores[score_type][score_subtype], dim=1).cpu()
        xs, ys, ypreds = torch.cat(xs).cpu(), torch.cat(ys).cpu(), torch.cat(ypreds).cpu()
        data_dump = {'scores': scores, 'x': xs, 'y': ys, 'ypred': ypreds}
        pickle.dump(data_dump, open('viz_scores_nba_{}{}.pkl'.format(args.tag, 'attr' if attribute_dataset else ''), 'wb'))


    th = get_th(opts, model)
    print('\n[{}] Epoch {} th: {}'.format(loader.dataset.mode.upper(), epoch, th))
    print('[{}] minADE/minFDE (1.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_1'] / avg_meter['counter'], avg_meter['fde_1'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (2.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_2'] / avg_meter['counter'], avg_meter['fde_2'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (3.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_3'] / avg_meter['counter'], avg_meter['fde_3'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (4.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_4'] / avg_meter['counter'], avg_meter['fde_4'] / avg_meter['counter']))
    if measure_flops: print('[{}] model FLOPs for one batch (size {}): {:.3f}'.format(loader.dataset.mode.upper(), batch_size, flops_per_batch))
    return avg_meter['fde_4'] / avg_meter['counter'], avg_meter['ade_4'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='nba', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_nba_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--tag', type=str, default="", help='log tag add-on to folder name')
    parser.add_argument("--test", action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
