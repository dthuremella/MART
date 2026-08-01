import os
import random
import argparse

from csv import writer

import torch
import numpy as np

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import MART
from loaders.dataloader_sdd import TrajectoryDataset, use_kalman
import time
import pickle

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

def my_collate(batch):
    '''
    Pads batch of variable length
    '''
    batch_x = []
    batch_y = []
    for t in batch:
        x, y = t
        batch_x.append(x)
        batch_y.append(y)

    ## pad sequences with zeros  
    batch_x_padded = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
    batch_y_padded = torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True)

    ## add mask
    pad_ones = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=1)
    mask = (batch_x_padded - pad_ones) + 1    ## ones for where there's values
    batch_x_mask = torch.cat((batch_x_padded, mask[:,:,:,:1]), dim=-1)
    return batch_x_mask, batch_y_padded

def my_collate_kalman(batch):
    '''
    Pads batch of variable length
    '''
    batch_x = []
    batch_y = []
    kalman_score = []
    for t in batch:
        x, y, k = t
        batch_x.append(x)
        batch_y.append(y)
        kalman_score.append(torch.tensor(k))
    ## pad sequences with zeros  
    batch_x_padded = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
    batch_y_padded = torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True)
    kalman_score_padded = torch.nn.utils.rnn.pad_sequence(kalman_score, batch_first=True)

    ## add mask
    pad_ones = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=1)
    mask = (batch_x_padded - pad_ones) + 1    ## ones for where there's values
    batch_x_mask = torch.cat((batch_x_padded, mask[:,:,:,:1]), dim=-1)
    return batch_x_mask, batch_y_padded, kalman_score_padded

def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)

    print('[INFO] The seed is:', seed)

    if not args.test:
        dataset_train = TrajectoryDataset(mode='train', scale=opts.scale, inputs=opts.inputs)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opts.batch_size, collate_fn=my_collate_kalman if use_kalman else my_collate, shuffle=True, num_workers=8, drop_last=True)
        
    dataset_test = TrajectoryDataset(mode='test', scale=opts.scale, inputs=opts.inputs)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, collate_fn=my_collate if (args.test or not use_kalman) else my_collate_kalman, shuffle=False, num_workers=8)

    if 'reported' in args.tag:
        opts.inputs = ['vel_x', 'vel_y']

    model = MART(opts).cuda()
    print('[INFO] Model params: {}'.format(sum(p.numel() for p in model.parameters())))
    if measure_flops:
        register_flop_hooks(model)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)
    if learnable_targets:
        target_params = [p for n, p in model.named_parameters() if 'target_logits' in n]
        optimizer_target = optim.Adam(target_params, lr=1e-3)  # can tune this lr separately

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
        ade, fde = test(model_ckpt['epoch'], model, loader_test)
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

        # run train again to update the target parameter
        if learnable_targets:
            train(epoch, model, optimizer_target, loader_train, train_EM_targets=True)

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
        print('[INFO] Best {} Loss: {:.5f} \t Best ade: {:.5f} \t Best epoch {}\n'.format('TEST', best_val_loss, best_ade, best_epoch))

        file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch) + '.pth')
        if epoch > 0:
            remove_file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch - 1) + '.pth')
            os.system('rm ' + remove_file_path)
            
        torch.save(state, file_path)
        
        if opts.scheduler_type is not None:
            scheduler.step()


def train(epoch, model, optimizer, loader, train_EM_targets=False):
    model.train()
    avg_meter = {'epoch': epoch, 'loss': 0, 'counter': 0}
    loader_len = len(loader)
    batch_count, divider = 0, 0
    is_first_loss = True

    if train_EM_targets:
        # Set the flag on all gates so forward() knows which KL direction to use
        for module in model.modules():
            if isinstance(module, GSoftmaxHarmonicGate):
                module.update_targets = True

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        batch_count += 1
        divider += 1
        
        kalman_score = None
        if use_kalman:
            x_abs, y, kalman = data
            x_abs, y, kalman = x_abs.cuda(), y.cuda(), kalman.cuda()
            if force_kalman or contrast_compress_embedding: 
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
            cur_pos = x_abs[:, :, [-1], :2].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos
            
        y = y[:, :, None, :, :]
        
        mask = x_abs[:,:,0,-1]
        total_loss = torch.sum(torch.min(      # minADE
                            torch.mean(torch.norm(y_pred - y, dim=-1), dim=3),
                            dim=2)[0] * mask    # mask out loss for invalid
                        ) 
        if harmonic_bias_loss is not None:
            total_loss += harmonic_bias_loss * avg_expert_idx

        avg_meter['loss'] += total_loss.item() * batch_size * num_agents
        avg_meter['counter'] += (batch_size * num_agents)

        if is_first_loss:
            loss = total_loss
            is_first_loss = False
        else:
            loss += total_loss

        if batch_count % opts.batch_size == 0 or i == loader_len - 1:
            loss = loss / divider
            is_first_loss = True
            
            loss.backward()
            if opts.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
                
            optimizer.step()

        if i % 100 == 0:
            if i != 0 and measure_flops:
                print('[{}][{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Flops: {} | LR: {}'
                    .format(args.dataset.upper(), 'TRAIN', epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), flops_per_batch, optimizer.param_groups[0]['lr']))
            else:
                th = get_th(opts, model)
                print('[{}][{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Threshold: {} | LR: {}'
                    .format(args.dataset.upper(), 'TRAIN', epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), th, optimizer.param_groups[0]['lr']))
    return avg_meter['loss'] / avg_meter['counter']


def test(epoch, model, loader):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade': 0, 'fde': 0, 'counter': 0}
    if viz:
        scores = {}
        for score_type in ['gate_score', 'top_k_idx']:
            scores[score_type] = {'pair_n': [], 'pair_e': [], 'group_n': [], 'group_e': []}
        xs, ys, ypreds = [], [], []

    t0 = time.time()
    with torch.no_grad():
        batch_count = 0
        for _, data in enumerate(loader):
            batch_count += 1
            if (use_kalman and not args.test):
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
                if measure_flops:
                    reset_flop_counts(model)

            dims = 2 if 'reported' in args.tag else 3
            y_pred, _, score = model(x_abs[...,:dims], x_rel[...,:dims])

            if batch_count == 2:
                if measure_flops:
                    flops_per_batch = get_total_flops(model)
                    if args.test:
                        print_flops(model)
                        import sys; sys.exit(0)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1], :2].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            if viz:
                x_flat = x_abs.flatten(0,1)
                mask = x_flat[:,-1,-1].bool() # most current timestep needs to be valid to be counted
                xs.append(x_flat[mask])
                ys.append(y.flatten(0,1).clone()[mask])
                ypreds.append(y_pred.flatten(0,1).clone()[mask])
                if moe_e or moe_n:
                    for score_type in ['gate_score', 'top_k_idx']:
                        for score_subtype in ['pair_n', 'group_n', 'group_e']:
                            if score[score_type][score_subtype][0] is None: continue
                            padded_scores = torch.stack(score[score_type][score_subtype])
                            padded_scores = padded_scores.permute(1,2,0,3)
                            unpadded_scores = padded_scores.flatten(0,1).clone()[mask]
                            unpadded_scores = unpadded_scores.permute(1,0,2)
                            scores[score_type][score_subtype].append(unpadded_scores)

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]
            
            mask = np.array(x_abs[:,:,0,-1].cpu())
            ade = np.sum(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2) * mask)
            fde = np.sum(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2) * mask)
                        
            avg_meter['ade'] += ade
            avg_meter['fde'] += fde
            
            avg_meter['counter'] += mask.sum()
    
    t1 = time.time()
    print('INFO: Time taken for inference is :', t1 - t0)

    if viz:
        target_logits_final = {}
        if learnable_targets:
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

        if moe_e or moe_n:
            for score_type in ['gate_score', 'top_k_idx']:
                for score_subtype in ['pair_n', 'pair_e', 'group_n', 'group_e']:
                    if len(scores[score_type][score_subtype]) == 0: continue
                    scores[score_type][score_subtype] = torch.cat(scores[score_type][score_subtype], dim=1).cpu()
        xs, ys, ypreds = torch.cat(xs).cpu(), torch.cat(ys).cpu(), torch.cat(ypreds).cpu()
        data_dump = {'scores': scores, 'x': xs, 'y': ys, 'ypred': ypreds, 'target_logits_final': target_logits_final}
        pickle.dump(data_dump, open('viz_scores_sdd_{}.pkl'.format(args.tag), 'wb'))

    avg_meter['ade'] /= opts.scale
    avg_meter['fde'] /= opts.scale
    
    th = get_th(opts, model)
    print('\n[{}][{}] Epoch {} th: {}'.format(args.dataset.upper(), 'TEST', epoch, th))
    print('[{}][{}] minADE/minFDE: {:.2f}/{:.2f}'.format(args.dataset.upper(), 'TEST', avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    if measure_flops: print('[{}] model FLOPs for one batch (size {}): {:.3f}'.format(args.dataset.upper(), batch_size, flops_per_batch))
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='sdd', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_sdd_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument("--test", action='store_true')
    parser.add_argument('--tag', type=str, default="", help='log tag add-on to folder name')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    # if args.test:
    #     opts.batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()