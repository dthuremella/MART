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
from loaders.dataloader_sdd import TrajectoryDataset
import pickle
from models.moe import deepseek_lb, K, NUM_EXPERTS
# python main_sdd.py --config ./configs/mart_sdd.yaml --gpu 1 --tag div8_top2_zloss

load_balance_layer_only = None #3 # set to None to do all layers

load_balance = False
load_balance_loss_only = False
router_z_loss = False 
clip_router_grad = False

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
        loader_train = torch.utils.data.DataLoader(dataset_train, collate_fn=my_collate,
                                                    batch_size=opts.batch_size, num_workers=8, 
                                                    shuffle=True, drop_last=True)
        
    dataset_test = TrajectoryDataset(mode='test', scale=opts.scale, inputs=opts.inputs)
    loader_test = torch.utils.data.DataLoader(dataset_test, collate_fn=my_collate,
                                                batch_size=opts.batch_size, num_workers=8,
                                                shuffle=False)

    model = MART(opts).cuda()
    print('[INFO] Model params: {}'.format(sum(p.numel() for p in model.parameters())))

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
        ade, fde = test(model_ckpt['epoch'], model, loader_test)
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('./results', '{}_result.csv'.format(args.dataset)), 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow([os.path.basename(args.config).split('.')[0], ade, fde])
        exit()

    results = {'epochs': [], 'test_losses': [], 'test_ades': [], 'train_losses': []}
    best_val_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    print('[INFO] The seed is :',seed)
    
    results_path = os.path.join(model_save_dir, 'sdd_results.pkl')
    lb_log = {}
    z_log = {}
    for k in ['pair_n', 'pair_e', 'group_n', 'group_e']:
        lb_log[k] = {}
        z_log[k] = {}
        for i in range(4): 
            lb_log[k][i] = []
            z_log[k][i] = []

    for epoch in range(0, opts.num_epochs):
        train_loss = train(epoch, model, optimizer, loader_train, lb_log=lb_log, z_log=z_log)
        test_loss, ade = test(epoch, model, loader_test)

        results['epochs'].append(epoch)
        results['test_losses'].append(test_loss.item())
        results['test_ades'].append(ade.item())
        results['train_losses'].append(train_loss.item())
        results['lb_log'] = lb_log
        results['z_log'] = z_log
        pickle.dump(results, open(results_path, 'wb'))

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        # if test_loss < best_val_loss:
        if ade < best_ade:
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


def train(epoch, model, optimizer, loader, lb_log=None, z_log=None):
    model.train()
    avg_meter = {'epoch': epoch, 'loss': 0, 'counter': 0}
    loader_len = len(loader)
    batch_count, divider = 0, 0
    is_first_loss = True

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        batch_count += 1
        divider += 1
        
        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()        
        
        batch_size, num_agents, length, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]
        
        y_pred, score, logit = model(x_abs, x_rel, epoch=epoch)
        lb_loss = 0
        if load_balance:
            for k in score:
                score[k] = torch.stack(score[k])

                # load balancing loss
                alpha = 1
                maxes, argmaxes = torch.max(score[k], -1)
                argmaxes = argmaxes.flatten(-2, -1)
                gating_scores_full = score[k].flatten(-3, -2)
                for u in range(score[k].shape[0]): # layers
                    if load_balance_layer_only is not None and u != load_balance_layer_only:
                        continue
                    for v in range(score[k].shape[1]):
                        unq, unq_counts = argmaxes[u][v].unique(return_counts=True)
                        counts = torch.zeros(score[k].shape[-1]).long().cuda()
                        counts[unq] = unq_counts
                        fi = counts.float() / argmaxes.shape[-1]
                        pi = torch.sum(gating_scores_full[u][v], dim=-2) / argmaxes.shape[-1]
                        loss_load_balance = alpha * pi.shape[0] * torch.dot(fi, pi)
                        lb_loss += loss_load_balance
                        if lb_log is not None: lb_log[k][u].append(loss_load_balance.item())
            lb_loss /= (len(score) * score[k].shape[0] * score[k].shape[1])
        z_loss = 0
        if router_z_loss:
            for k in logit:
                logit[k] = torch.stack(logit[k])

                # router z loss
                alpha = 0.01
                for u in range(logit[k].shape[0]):
                    for v in range(logit[k].shape[1]):
                        exp = torch.exp(logit[k][u][v])
                        sum_over_num_experts = torch.sum(exp, dim=-1)
                        squared_log = torch.pow(torch.log(sum_over_num_experts), 2)
                        z_logit_term = alpha * torch.mean(squared_log)
                        z_loss += z_logit_term
                        if z_log is not None: z_log[k][u].append(z_logit_term.item())
            z_loss /= (len(logit) * logit[k].shape[0] * logit[k].shape[1])

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1], :2].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos
            
        y = y[:, :, None, :, :]
        
        mask = x_abs[:,:,0,-1]
        if load_balance_loss_only:
            total_loss = lb_loss + z_loss
        else:
            total_loss = torch.sum(torch.min(      # minADE
                                torch.mean(torch.norm(y_pred - y, dim=-1), dim=3),
                                dim=2)[0] * mask    # mask out loss for invalid
                            ) 
            total_loss += (lb_loss + z_loss)
        
        avg_meter['loss'] += total_loss.item()
        avg_meter['counter'] += mask.sum()

        if is_first_loss:
            loss = total_loss
            is_first_loss = False
        else:
            loss += total_loss

        if batch_count % opts.batch_size == 0: # or i == loader_len - 1:
            loss = loss / divider
            is_first_loss = True
            
            loss.backward()
            if opts.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
            elif clip_router_grad:
                params_to_clip = []
                for pair_encoder in model.pair_encoders:
                    for layer in pair_encoder.layers:
                        params_to_clip.extend(list(layer.linear_net_n.gate.parameters()))
                        params_to_clip.extend(list(layer.linear_net2_e.gate.parameters()))
                for hyper_encoder in model.hyper_encoders:
                    for layer in hyper_encoder.layers:
                        params_to_clip.extend(list(layer.linear_net_n.gate.parameters()))
                        params_to_clip.extend(list(layer.linear_net2_e.gate.parameters()))
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)

                
            optimizer.step()

            # train deepseek expert biases layer
            if deepseek_lb:
                update_rate = 1e-2
                for k in score: # these are the 4 different score types (pair_n, pair_e, group_n, group_e)
                    for u in range(len(score[k])): # these are the layer numbers   
                        if load_balance_layer_only is not None and u != load_balance_layer_only:
                            continue              
                        for v in range(len(score[k][u])): # there's only 1 dimension of this
                            topk_scores, topk_idx = torch.topk(score[k][u][v], K, dim=-1, sorted=False)
                            expert_counts = torch.bincount(topk_idx.flatten(), minlength=NUM_EXPERTS) 
                            
                            ### logging
                            if k == 'pair_n':
                                biases = model.pair_encoders[u].layers[v].linear_net_n.expert_biases.data
                            elif k == 'pair_e':
                                biases = model.pair_encoders[u].layers[v].linear_net2_e.expert_biases.data
                            elif k == 'group_n':
                                biases = model.hyper_encoders[u].layers[v].linear_net_n.expert_biases.data
                            elif k == 'group_e':
                                biases = model.hyper_encoders[u].layers[v].linear_net2_e.expert_biases.data
                            tens = torch.stack([biases, expert_counts], dim=-1).cpu()
                            lb_log[k][u].append(tens)  # to keep the logs aligned      
                            
                            avg_count = expert_counts.float().mean()
                            for i, count in enumerate(expert_counts):
                                # b_i = b_i + u + sign(e_i)
                                # note: this is \bar{c_i} - c_i, NOT c_i - \bar{c_i}, which will push the network to
                                # be maximally unbalanced. Really important to get this part right!!!
                                error = avg_count - count.float()
                                if k == 'pair_n':
                                    model.pair_encoders[u].layers[v].linear_net_n.expert_biases.data[i] += update_rate * torch.sign(error)
                                elif k == 'pair_e':
                                    model.pair_encoders[u].layers[v].linear_net2_e.expert_biases.data[i] += update_rate * torch.sign(error)
                                elif k == 'group_n':
                                    model.hyper_encoders[u].layers[v].linear_net_n.expert_biases.data[i] += update_rate * torch.sign(error)
                                elif k == 'group_e':
                                    model.hyper_encoders[u].layers[v].linear_net2_e.expert_biases.data[i] += update_rate * torch.sign(error)

                

            th = get_th(opts, model)
            print('[{}][{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Threshold: {} | LR: {}'
                  .format(args.dataset.upper(), 'TRAIN', epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), th, optimizer.param_groups[0]['lr']))
    return avg_meter['loss'] / avg_meter['counter']


def test(epoch, model, loader):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade': 0, 'fde': 0, 'counter': 0}
    
    with torch.no_grad():
        scores = {'pair_n': [], 'pair_e': [], 'group_n': [], 'group_e': []}
        xs, ys, ypreds = [], [], []
        for _, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()        
            
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]
            
            y_pred, score, logit = model(x_abs, x_rel)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1], :2].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            for k in score:
                score[k] = torch.stack(score[k]).cpu()
                scores[k].append(score[k])
            xs.append(x_abs.cpu())
            ys.append(y.cpu())
            ypreds.append(y_pred.cpu())

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]
            
            mask = np.array(x_abs[:,:,0,-1].cpu())
            ade = np.sum(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2) * mask)
            fde = np.sum(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2) * mask)
                        
            avg_meter['ade'] += ade
            avg_meter['fde'] += fde
            
            avg_meter['counter'] += mask.sum()
    
    avg_meter['ade'] /= opts.scale
    avg_meter['fde'] /= opts.scale

    # for k in scores:
    #     scores[k] = torch.cat(scores[k], dim=2).cpu()
    # xs, ys, ypreds = torch.cat(xs).cpu(), torch.cat(ys).cpu(), torch.cat(ypreds).cpu()
    data_dump = {'scores': scores, 'x': xs, 'y': ys, 'ypred': ypreds}
    pickle.dump(data_dump, open('viz_scores_sdd_{}.pkl'.format(args.tag), 'wb'))

    
    th = get_th(opts, model)
    print('\n[{}][{}] Epoch {} th: {}'.format(args.dataset.upper(), 'TEST', epoch, th))
    print('[{}][{}] minADE/minFDE: {:.2f}/{:.2f}'.format(args.dataset.upper(), 'TEST', avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='sdd', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_sdd_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--tag', type=str, default="", help='log tag add-on to folder name')
    parser.add_argument("--test", action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    if args.test:
        opts.batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()