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
from loaders.dataloader_eth import TrajectoryDataset
import pickle

arg_load_balance = True

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

    data_root = os.path.join('./datasets/ethucy', args.dataset)

    if not args.test:
        dataset_train = TrajectoryDataset(args, os.path.join(data_root, 'train'), obs_len=opts.past_length, pred_len=opts.future_length, skip=1)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opts.batch_size, collate_fn=my_collate,
                                shuffle=True, num_workers=8, drop_last=True)
        
    dataset_test = TrajectoryDataset(args, os.path.join(data_root, 'test'), obs_len=opts.past_length, pred_len=opts.future_length, skip=1)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, collate_fn=my_collate,
                                shuffle=False, num_workers=8)

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
    
    results_path = os.path.join(model_save_dir, '{}_results.pkl'.format(args.dataset))
    for epoch in range(0, opts.num_epochs):
        train_loss = train(epoch, model, optimizer, loader_train)
        test_loss, ade = test(epoch, model, loader_test)
        results['epochs'].append(epoch)
        results['test_losses'].append(test_loss)
        results['test_ades'].append(ade)
        results['train_losses'].append(train_loss)
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


def train(epoch, model, optimizer, loader):
    model.train()
    avg_meter = {'epoch': epoch, 'loss': 0, 'counter': 0}
    loader_len = len(loader)
    batch_count, divider = 0, 0
    is_first_loss = True

    scores = {'pair_n': [], 'pair_e': [], 'group_n': [], 'group_e': []}
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
        
        y_pred, score, logits = model(x_abs, x_rel, epoch=epoch)
        lb_loss = 0
        if args.load_balance:
            print('[INFO] Load balancing loss enabled')
            for k in score:
                score[k] = torch.stack(score[k])
                scores[k].append(score[k])

                # load balancing loss
                alpha = 0.01
                maxes, argmaxes = torch.max(score[k], -1)
                argmaxes = argmaxes.flatten(-2, -1)
                gating_scores_full = score[k].flatten(-3, -2)
                for u in range(score[k].shape[0]):
                    for v in range(score[k].shape[1]):
                        unq, counts = argmaxes[u][v].unique(return_counts=True)
                        fi = counts.float() / argmaxes.shape[-1]
                        pi = torch.sum(gating_scores_full[u][v], dim=-2) / argmaxes.shape[-1]
                        fi = torch.cat((fi, torch.zeros(pi.shape[0]-fi.shape[0]).cuda())) # in case all aren't filled
                        loss_load_balance = alpha * pi.shape[0] * torch.dot(fi, pi)
                        lb_loss += loss_load_balance
            lb_loss /= (len(score) * score[k].shape[0] * score[k].shape[1])

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1], :2].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos
            
        y = y[:, :, None, :, :]
        
        mask = x_abs[:,:,0,-1]
        total_loss = torch.sum(torch.min(torch.mean(torch.norm(y_pred - y, dim=-1), dim=3), dim=2)[0] * mask) # for all agents
        total_loss += lb_loss
        
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
                
            optimizer.step()

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
            
            y_pred, score, logits = model(x_abs, x_rel)

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

    # for k in scores:
    #     scores[k] = torch.cat(scores[k], dim=2).cpu()
    # xs, ys, ypreds = torch.cat(xs).cpu(), torch.cat(ys).cpu(), torch.cat(ypreds).cpu()
    data_dump = {'scores': scores, 'x': xs, 'y': ys, 'ypred': ypreds}
    pickle.dump(data_dump, open('viz_scores_{}_{}.pkl'.format(args.dataset, args.tag), 'wb'))

    th = get_th(opts, model)
    print('\n[{}][{}] Epoch {} th: {}'.format(args.dataset.upper(), 'TEST', epoch, th))
    print('[{}][{}] minADE/minFDE: {:.2f}/{:.2f}'.format(args.dataset.upper(), 'TEST', avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='eth', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_eth_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--tag', type=str, default="", help='log tag add-on to folder name')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--load_balance", action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    if args.dataset == 'eth' or args.dataset == 'univ':
        opts.lr *= 1
    elif args.dataset == 'zara1' or args.dataset == 'zara2':
        opts.lr *= 1.2
    else:
        opts.lr *= 1.8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.test:
        opts.batch_size = 1

    main()