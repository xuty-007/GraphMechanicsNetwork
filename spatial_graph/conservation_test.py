import argparse
import torch
import torch.utils.data
from n_body_system.dataset_nbody import NBodyDataset, NBodyMStickDataset, NewNBodyMStickDataset
from n_body_system.model import GNN, EGNN, Baseline, Linear, EGNN_vel, Linear_dynamics, RF_vel, GVN
import os
from torch import nn, optim
import json
import time

import random
import numpy as np
from scipy.linalg import qr

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--sweep_training', type=int, default=0, metavar='N',
                    help='0 nor sweep, 1 sweep, 2 sweep small')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--n_isolated', type=int, default=5,
                    help='Number of isolated balls.')
parser.add_argument('--n_stick', type=int, default=0,
                    help='Number of sticks.')
parser.add_argument('--n_hinge', type=int, default=0,
                    help='Number of hinges.')
parser.add_argument('--data_dir', type=str, default='spatial_graph/n_body_system/new_dataset/data',
                    help='Data directory.')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument("--config_by_file", default=False, action="store_true", )

time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
if args.config_by_file:
    job_param_path = './job_param.json'
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        args.exp_name = hyper_params["exp_name"]
        args.batch_size = hyper_params["batch_size"]
        args.epochs = hyper_params["epochs"]
        args.no_cuda = hyper_params["no_cuda"]
        args.seed = hyper_params["seed"]
        args.lr = hyper_params["lr"]
        args.nf = hyper_params["nf"]
        args.model = hyper_params["model"]
        args.attention = hyper_params["attention"]
        args.n_layers = hyper_params["n_layers"]
        args.degree = hyper_params["degree"]
        args.max_training_samples = hyper_params["max_training_samples"]
        # Do not necessary in practice.
        #args.dataset = hyper_params["dataset"]
        args.data_dir = hyper_params["data_dir"]
        args.weight_decay = hyper_params["weight_decay"]
        args.norm_diff = hyper_params["norm_diff"]
        args.tanh = hyper_params["tanh"]
        args.dropout = hyper_params["dropout"]

        args.n_isolated = hyper_params["n_isolated"]
        args.n_stick = hyper_params["n_stick"]
        args.n_hinge = hyper_params["n_hinge"]

args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

# torch.autograd.set_detect_anomaly(True)

def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    n_isolated, n_stick, n_hinge = args.n_isolated, args.n_stick, args.n_hinge
    args.batch_size = 5

    dataset_test = NewNBodyMStickDataset(partition='test', dataset_name="nbody_small", n_isolated=n_isolated,
                                         n_stick=n_stick, n_hinge=n_hinge, data_dir=args.data_dir)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    if args.model == 'gnn':
        model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    elif args.model == 'egnn_vel':
        model = EGNN_vel(in_node_nf=1, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                         recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    elif args.model == 'egnn_vel_cons':
        model = EGNN_vel(in_node_nf=1, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                         recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    elif args.model == 'gvn':
        model = GVN(in_node_nf=1, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                    recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh, dropout=args.dropout)
    elif args.model == 'baseline':
        model = Baseline()
    elif args.model == 'linear_vel':
        model = Linear_dynamics(device=device)
    elif args.model == 'linear':
        model = Linear(6, 3, device=device)
    elif args.model == 'rf_vel':
        model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2 + 1, device=device, act_fn=nn.SiLU(), n_layers=args.n_layers)
    elif args.model == 'se3_transformer' or args.model == 'tfn':
        from n_body_system.se3_dynamics.dynamics import OurDynamics as SE3_Transformer
        model = SE3_Transformer(n_particles=5, n_dimesnion=3, nf=int(args.nf/args.degree), n_layers=args.n_layers,
                                model=args.model, num_degrees=args.degree, div=1)
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        raise Exception("Wrong model specified")

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'losess': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_test)


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0, 'loss_vel': 0, 'reg_loss': 0}
    # res_energy = {'gt': 0, 'method': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data, cfg = data[:-1], data[-1]
        data = [d.to(device) for d in data]
        # data, mask = data[:-1], data[-1]
        # data, loc_end, vel_end = data[:-2], data[-2], data[-1]
        data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        loc, vel, edge_attr, charges, loc_end, vel_end = data
        # loc, vel, edge_attr, charges = data


        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        cfg = loader.dataset.get_cfg(batch_size, n_nodes, cfg)
        cfg = {_: cfg[_].to(device) for _ in cfg}

        optimizer.zero_grad()

        Q = np.random.randn(3, 3)
        Q = qr(Q)[0]
        Q = torch.from_numpy(np.array(Q)).float().cuda()
        new_loc, new_vel = torch.matmul(loc, Q), torch.matmul(vel, Q)
        model.eval()

        def inference(_loc, _vel, edge_attr):

            if args.model == 'gnn':
                nodes = torch.cat([_loc, _vel], dim=1)
                loc_pred = model(nodes, edges, edge_attr)
            elif args.model == 'egnn':
                nodes = torch.ones(_loc.size(0), 1).to(device)  # all input nodes are set to 1
                rows, cols = edges
                loc_dist = torch.sum((_loc[rows] - _loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                vel_attr = get_velocity_attr(_loc, _vel, rows, cols).detach()
                edge_attr = torch.cat([edge_attr, loc_dist, vel_attr], 1).detach()  # concatenate all edge properties
                loc_pred = model(nodes, _loc.detach(), edges, edge_attr)
            elif args.model == 'egnn_vel' or args.model == 'egnn_vel_cons':
                nodes = torch.sqrt(torch.sum(_vel ** 2, dim=1)).unsqueeze(1).detach()
                rows, cols = edges
                loc_dist = torch.sum((_loc[rows] - _loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
                loc_pred = model(nodes, _loc.detach(), edges, _vel, edge_attr)
            elif args.model == 'gvn':
                nodes = torch.sqrt(torch.sum(_vel ** 2, dim=1)).unsqueeze(1).detach()
                rows, cols = edges
                loc_dist = torch.sum((_loc[rows] - _loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
                loc_pred, vel_pred = model(nodes, _loc.detach(), edges, _vel, cfg, edge_attr)
            elif args.model == 'baseline':
                backprop = False
                loc_pred = model(_loc)
            elif args.model == 'linear':
                loc_pred = model(torch.cat([_loc, _vel], dim=1))
            elif args.model == 'linear_vel':
                loc_pred = model(_loc, _vel)
            elif args.model == 'se3_transformer' or args.model == 'tfn':
                loc_pred = model(_loc, _vel, charges)
            elif args.model == 'rf_vel':
                rows, cols = edges
                vel_norm = torch.sqrt(torch.sum(_vel ** 2, dim=1).unsqueeze(1)).detach()
                loc_dist = torch.sum((_loc[rows] - _loc[cols]) ** 2, 1).unsqueeze(1)
                edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
                loc_pred = model(vel_norm, _loc.detach(), edges, _vel, edge_attr)
            else:
                raise Exception("Wrong model")

            return loc_pred, None

        loc1, _x1 = inference(loc, vel, edge_attr)
        loc2, _x2 = inference(new_loc, new_vel, edge_attr)

        tranf_loc1 = torch.matmul(loc1, Q)
        tranf_x1 = _x1
        print('Distance:', torch.sum(torch.abs(tranf_loc1 - loc2)))
        exit(0)


def main_sweep():
    training_samples = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25000, 50000]
    n_epochs = [2000, 2000, 4000, 5000, 8000, 10000, 8000, 6000, 4000, 2000]
    if args.model == 'egnn_vel':
        n_epochs = [4000, 4000, 2000, 2000, 2000, 1500, 1500, 1500, 1000, 1000]  # up to the 5th updated
    elif args.model == 'kholer_vel':
        n_epochs = [8000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 4000, 2000]  # up to the 5th

    if args.sweep_training == 2:
        training_samples = training_samples[0:5]
        n_epochs = n_epochs[0:5]
    elif args.sweep_training == 3:
        training_samples = training_samples[6:]
        n_epochs = n_epochs[6:]
    elif args.sweep_training == 4:
        training_samples = training_samples[8:]
        n_epochs = n_epochs[8:]

    results = {'tr_samples': [], 'test_loss': [], 'best_epochs': []}
    for epochs, tr_samples in zip(n_epochs, training_samples):
        args.epochs = epochs
        args.max_training_samples = tr_samples
        args.test_interval = max(int(10000/tr_samples), 1)
        best_val_loss, best_test_loss, best_epoch = main()
        results['tr_samples'].append(tr_samples)
        results['best_epochs'].append(best_epoch)
        results['test_loss'].append(best_test_loss)
        print("\n####### Results #######")
        print(results)
        print("Results for %d epochs and %d # training samples \n" % (epochs, tr_samples))


if __name__ == "__main__":
    if args.sweep_training:
        main_sweep()
    else:
        best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_apoch = %d" % best_epoch)
    print("best_train = %.6f, best_val = %.6f, best_test = %.6f, best_apoch = %d" % (best_train_loss, best_val_loss, best_test_loss, best_epoch))





