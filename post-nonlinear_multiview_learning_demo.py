from __future__ import print_function
import argparse
import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib
import scipy.linalg as spalg

# Argument parser
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_views", default=2, help="Number of views", type=int)

    parser.add_argument("--latent_dim", default=2, help="Dimensionality of shared", type=int)

    parser.add_argument("--batch_size", default=1000, help="Batch size", type=int)

    parser.add_argument("--num_epochs", default=100, help="Number of epochs", type=int)

    parser.add_argument("--inner_iters", default=100, help="Number of inner iterations", type=int)

    parser.add_argument("--learning_rate", default=1e-3, help="Learning rate", type=float)

    parser.add_argument("--_lambda", default=1e-3, help="Value of lambda", type=float)

    parser.add_argument("--model_file", default='best_model_multiview.pth',
            help="Filename for best model saving", type=str)

    # Structure for encoder and decoder network
    parser.add_argument("--f_num_layers", default=1, help="Number of layers for f", type=int)

    parser.add_argument("--f_hidden_size", default=128, help="Number of hidden neurons for f",
            type=int)

    parser.add_argument("--g_num_layers", default=1, help="Number of layers for g", type=int)

    parser.add_argument("--g_hidden_size", default=128, help="Number of hidden neurons for g",
            type=int)

    return parser

# Post-Nonlinear Multiview Model
class PNMM(nn.Module):
    #input_dims: array, each element is the feature size of the corresponding view
    #f_size: array, network structure of f_{NN} function
    #g_size: array, network structure of g_{NN} function
    #latent_d: scalar, the latent dimision
    def __init__(self, input_dims, f_size, g_size, latent_d):
        super().__init__()

        # encoding
        self.e_net = nn.ModuleList()
        # decoding
        self.d_net = nn.ModuleList()
        # record dimension
        self.dims = input_dims
        # latent dimension
        self.latent_d = latent_d

        # For each view, use the 1D convolutional network to implement the
        # element-wise neural network structure
        for i in self.dims:
            ### encoding network
            tmp = []
            # input layer
            tmp.append(nn.Conv1d(i, f_size[0]*i, 1, groups=i))
            tmp.append(nn.ReLU())

            # hidden layers
            for j in range(1,len(f_size)):
                tmp.append(nn.Conv1d(f_size[j-1]*i, f_size[j]*i, 1, groups=i))
                tmp.append(nn.ReLU())

            # output layer
            tmp.append(nn.Conv1d(f_size[-1]*i, i, 1, groups=i))

            # the f network for each view
            self.e_net.append(nn.Sequential(*tmp))

            # the B matrix for each view
            self.e_net.append(nn.Linear(i, self.latent_d, bias=False))

            ### decoding network
            tmp = []
            # input layer
            tmp.append(nn.Conv1d(i, g_size[0]*i, 1, groups=i))
            tmp.append(nn.ReLU())

            # hidden layers
            for j in range(1,len(g_size)):
                tmp.append(nn.Conv1d(g_size[j-1]*i, g_size[j]*i, 1, groups=i))
                tmp.append(nn.ReLU())

            # output layer
            tmp.append(nn.Conv1d(g_size[-1]*i, i, 1, groups=i))

            # the g_{NN} network for each view
            self.d_net.append(nn.Sequential(*tmp))


    # encoding function
    def encode(self, x):
        # the output of the network
        f_x=[]
        # the output after multiplied by B
        B_f_x=[]

        start = 0
        # for each view
        for i in range(len(self.dims)):
            tmp = self.e_net[2*i](x[:, start:start+self.dims[i]].view(-1, self.dims[i], 1))
            f_x.append(tmp.squeeze())
            B_f_x.append(self.e_net[2*i+1](tmp.squeeze()))

            start += self.dims[i]

        return torch.cat(B_f_x, 1), torch.cat(f_x, 1)

    # decoding function
    def decode(self, x):
        g_f_x=[]
        start = 0
        # for each view
        for i in range(len(self.dims)):
            tmp = self.d_net[i](x[:, start:start+self.dims[i]].view(-1, self.dims[i], 1))
            g_f_x.append(tmp.squeeze())

            start += self.dims[i]

        return torch.cat(g_f_x, 1)


# Loss function
def loss_function(fx, gx, x, U_bat, num_view):

    # the MSE loss
    l = nn.MSELoss(reduction='sum')

    # encoding loss
    cca_err = l(fx, U_bat.repeat(1, num_view))

    # decoding loss
    recons_err = l(gx, x)

    return cca_err, recons_err


# Training function
def train(model, optimizer, train_loader, device, epoch, num_view, U_all, _lambda):
    model.train()
    train_loss = 0
    cca_err=0
    recons_err=0
    for batch_idx, (data, _, idxes) in enumerate(train_loader):
        data = data.to(device)

        # Forward
        fy, fx = model.encode(data)

        # Corresponding batch of U
        U_bat = U_all[idxes, :]

        # Reconstruction
        g_fx = model.decode(fx)

        # Compute the loss
        c_e, r_e = loss_function(fy, g_fx, data, U_bat, num_view)
        loss = c_e + _lambda*r_e
        train_loss += loss.item()
        cca_err += c_e.item()
        recons_err += r_e.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('====> Epoch: {} total loss: {:.4f}, c = {:.4f}, r = {:.4f}'.format(epoch, train_loss, cca_err, recons_err))

    return cca_err, recons_err

# Normalize for better visualization
def visual_norm(x):
    bound = 10
    x = x-np.amin(x)
    x = x/np.amax(x)*bound

    return x

# Dataset class
class ViewDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.data_len = data_tensor.shape[0]

    def __getitem__(self, index):
        return (self.data[index], 0, index)

    def __len__(self):
        return self.data_len

def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(12)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_file = 'synthetic_data_2view.mat'
    in_data = sio.loadmat(data_file)

    view1 = in_data['view1']
    view2 = in_data['view2']
    base = in_data['shared']
    base_q = in_data['Q']
    mix1 = in_data['mix1']
    mix2 = in_data['mix2']

    mix = np.hstack([mix1, mix2])

    dims=[]
    dims.append(view1.shape[1])
    dims.append(view2.shape[1])

    views = np.hstack([view1, view2])
    views = torch.tensor(views)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PNMM(dims, [args.f_hidden_size]*args.f_num_layers,
            [args.g_hidden_size]*args.g_num_layers, args.latent_dim)
    model = model.to(device)
    model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    data_set=ViewDataset(views)

    train_loader = torch.utils.data.DataLoader(data_set,
        batch_size=args.batch_size, shuffle=True)

    train_eval_loader = torch.utils.data.DataLoader(data_set,
        batch_size=args.batch_size, shuffle=False)

    # Training starts
    encode_err=[]
    decode_err=[]
    range_dist=[]

    best_train = float('inf')
    for epoch in range(1, args.num_epochs + 1):
        # Fix f() and g()
        model.eval()
        F = []
        with torch.no_grad():
            for batch_idx, (data, _, _) in enumerate(train_eval_loader):
                data = data.to(device)

                # Forward
                bfx, _ = model.encode(data)
                F.append(bfx)

            F = torch.cat(F, 0)
            F = F - torch.mean(F, 0, True)

            tmp = []
            for i in range(args.num_views):
                tmp.append(F[:,i*args.latent_dim:(i+1)*args.latent_dim])

            sum_f = torch.stack(tmp, dim=2)
            # Compute SVD to generate U
            u, _, v = torch.svd(torch.sum(sum_f, dim=2))
            U_n = torch.mm(u, v.t())
            # Normalize by sample size
            U = U_n * U_n.shape[0]**0.5

            # Evaluate with subspace distance
            dist = np.sin(spalg.subspace_angles(base_q, U_n.cpu().numpy()))[0]
            range_dist.append(dist)

        # fix U
        for _ in range(args.inner_iters):
            e_e, d_e = train(model, optimizer, train_loader, device, epoch,
                    args.num_views, U, args._lambda)

            if e_e+d_e < best_train:
                print('Saving Model')
                torch.save(model.state_dict(), args.model_file)
                best_train = e_e+d_e

            encode_err.append(e_e)
            decode_err.append(d_e)


    # Load the best model
    model.load_state_dict(torch.load(args.model_file))
    model = model.to(device)
    model = model.double()

    # Evaluate the model
    with torch.no_grad():
        BF, F = [], []
        for batch_idx, (data, _, _) in enumerate(train_eval_loader):
            data = data.to(device)
            bf, f = model.encode(data)
            #bfx = tmp[0].cpu().numpy()
            #fx = tmp[1].cpu().numpy()
            BF.append(bf)
            F.append(f)

    BF = torch.cat(BF, 0).cpu().numpy()
    F = torch.cat(F, 0).cpu().numpy()

    # Plot subspace distance
    plt.plot(range_dist)
    plt.title('Subspace distance', fontsize=20)
    plt.show()

    # Scatter plot of the learned shared component
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    axs[0].scatter(BF[:,0], BF[:,1])
    axs[1].scatter(BF[:,2], BF[:,3])
    plt.show()

    # Plot the composite functions
    for i in range(mix.shape[1]):
        plt.scatter(mix[:,i], visual_norm(F[:,i]), marker='o', s=50,
            label='$\widehat{f}^{\ ('+str(i//dims[0]+1)+')}_'+str(i%dims[0]+1)+'\circ g^{('+str(i//dims[0]+1)+')}_'+str(i%dims[0]+1)+'$')
        plt.scatter(mix[:,i], visual_norm(views[:,i].numpy()), marker='^', s=50,
                label='$g^{('+str(i//dims[0]+1)+')}_'+str(i%dims[0]+1)+'$')
        plt.xlabel('input',fontsize=20,fontweight='bold')
        plt.ylabel('output',fontsize=20,fontweight='bold')
        plt.legend(fontsize=20)
        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
