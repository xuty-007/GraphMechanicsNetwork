import numpy as np
import torch
import random
import pickle as pkl


class NBodyDataset():
    """
    NBodyDataset

    """
    def __init__(self, partition='train', max_samples=1e8, dataset_name="se3_transformer", data_dir='spatial_graph'):
        self.partition = partition
        self.data_dir = data_dir
        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            # self.sufix += "_charged5_initvel1small"
            # TODO: change the ball number here
            ball_number = 5
            print('Loading {:d} balls'.format(ball_number))
            self.sufix += "_charged{:d}_initvel1small".format(ball_number)
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        loc = np.load(self.data_dir + '/' + 'n_body_system/dataset/loc_' + self.sufix + '.npy')
        vel = np.load(self.data_dir + '/' + 'n_body_system/dataset/vel_' + self.sufix + '.npy')
        edges = np.load(self.data_dir + '/' + 'n_body_system/dataset/edges_' + self.sufix + '.npy')
        charges = np.load(self.data_dir + '/' + 'n_body_system/dataset/charges_' + self.sufix + '.npy')

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        # swap n_nodes <--> batch_size and add nf dimension
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)
        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


class NBodyMStickDataset():
    """
    NBodyDataset

    """
    def __init__(self, partition='train', max_samples=1e8, dataset_name="se3_transformer", data_dir='spatial_graph',
                 stick_num=2, ball_number=5):
        self.partition = partition
        self.data_dir = data_dir
        # stick_num = 2
        self.stick_num = stick_num

        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
        # if self.partition == 'train':
        #     stick_num = 2
        #     self.stick_num = 2
        #     ball_number = 4
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            # self.sufix += "_charged5_initvel1small"
            # TODO: change the ball number here
            print('Loading {:d} balls {:d} sticks'.format(ball_number, stick_num))
            if stick_num == 0:
                # use the original dataset
                self.sufix += "_charged{:d}_initvel1small".format(ball_number)
            else:
                self.sufix += "_charged{:d}_{:d}_initvel1small".format(ball_number, stick_num)
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges, self.sticks = self.load()

    def load(self):
        loc = np.load(self.data_dir + '/' + 'n_body_system/dataset/loc_' + self.sufix + '.npy')
        vel = np.load(self.data_dir + '/' + 'n_body_system/dataset/vel_' + self.sufix + '.npy')
        edges = np.load(self.data_dir + '/' + 'n_body_system/dataset/edges_' + self.sufix + '.npy')
        charges = np.load(self.data_dir + '/' + 'n_body_system/dataset/charges_' + self.sufix + '.npy')
        try:
            sticks = np.load(self.data_dir + '/' + 'n_body_system/dataset/sticks_' + self.sufix + '.npy')
        except:
            print(self.data_dir + '/' + 'n_body_system/dataset/sticks_' + self.sufix + '.npy', 'Not Found! Skip!')
            sticks = None

        loc, vel, edge_attr, edges, charges, sticks = self.preprocess(loc, vel, edges, charges, sticks)
        return (loc, vel, edge_attr, charges), edges, sticks

    def preprocess(self, loc, vel, edges, charges, sticks):
        # cast to torch and swap n_nodes <--> n_features dimensions
        # convert stick [M, 2]
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edges = edges[:self.max_samples, ...]  # add here for better consistency
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # remove self loop
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]

        # swap n_nodes <--> batch_size and add nf dimension
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)  # [B, N*(N-1), 1]

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges), sticks

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges, self.sticks = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        if self.sticks is None:
            sticks = []
        else:
            sticks = self.sticks[i]
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)
        # concat stick indicator to edge_attr (for egnn_vel)
        edges = self.edges
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            for stick in sticks:
                if (row == stick[0] and col == stick[1]) or (row == stick[1] and col == stick[0]):
                    stick_ind[m] = 1
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)

        # initialize the connected-body matrix, with self loop  TODO: convert to mapping to subgraph later
        n_nodes = loc.shape[1]
        c_row, c_col = [_ for _ in range(n_nodes)], [_ for _ in range(n_nodes)]
        for stick in sticks:
            c_col[stick[0]] = stick[1]
            c_col[stick[1]] = stick[0]
        c_row, c_col = torch.from_numpy(np.array(c_row)), torch.from_numpy(np.array(c_col))

        # iso = np.ones(n_nodes)
        # for stick in sticks:
        #     iso[stick[0]] = 0
        #     iso[stick[1]] = 0
        #
        # cfg = {'Stick': []}
        # # for i in range(len(iso)):
        # #     if iso[i]:
        # #         cfg['Isolated'].append(i)
        # # cfg['Isolated'] = torch.from_numpy(np.array(cfg['Isolated'])).unsqueeze(-1)
        #
        # for stick in sticks:
        #     cfg['Stick'].append(stick)
        # cfg['Stick'] = torch.from_numpy(np.array(cfg['Stick']))

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T], vel[frame_T], c_row, c_col  # mask
        # return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T], vel[frame_T], cfg

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    def get_c_edges(self, batch_size, n_nodes, c_row, c_col):
        offset = torch.arange(batch_size) * n_nodes
        offset = offset.unsqueeze(-1).expand_as(c_row)
        c_row = c_row + offset
        c_col = c_col + offset
        return c_row.reshape(-1), c_col.reshape(-1)

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


class NewNBodyMStickDataset():
    """
    NBodyDataset

    """
    def __init__(self, partition='train', max_samples=1e8, dataset_name="se3_transformer",
                 data_dir='',
                 n_isolated=5, n_stick=0, n_hinge=0):
        self.partition = partition
        self.data_dir = data_dir
        self.n_isolated,  self.n_stick, self.n_hinge = n_isolated, n_stick, n_hinge

        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition

        self.dataset_name = dataset_name

        if dataset_name == "nbody_small":
            self.suffix += '_charged{:d}_{:d}_{:d}'.format(n_isolated, n_stick, n_hinge)
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges, self.cfg = self.load()

    def load(self):
        loc = np.load(self.data_dir + '/' + 'loc_' + self.suffix + '.npy')
        vel = np.load(self.data_dir + '/' + 'vel_' + self.suffix + '.npy')
        charges = np.load(self.data_dir + '/' + 'charges_' + self.suffix + '.npy')
        edges = np.load(self.data_dir + '/' + 'edges_' + self.suffix + '.npy')
        with open(self.data_dir + '/' + 'cfg_' + self.suffix + '.pkl', 'rb') as f:
            cfg = pkl.load(f)

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges, cfg

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        # convert stick [M, 2]
        loc, vel = torch.Tensor(loc), torch.Tensor(vel)  # remove transpose this time
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0: self.max_samples]
        edges = edges[: self.max_samples, ...]  # add here for better consistency
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # remove self loop
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]

        # swap n_nodes <--> batch_size and add nf dimension
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)  # [B, N*(N-1), 1]

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges, self.cfg = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)
        # concat stick indicator to edge_attr (for egnn_vel)
        edges = self.edges
        # initialize the configurations
        cfg = self.cfg[i]
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            if 'Stick' in cfg:
                for stick in cfg['Stick']:
                    if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
                    # if (row == stick[0] and col == stick[1]) or (row == stick[1] and col == stick[0]):
                        stick_ind[m] = 1
            if 'Hinge' in cfg:
                for hinge in cfg['Hinge']:
                    if (row, col) in [(hinge[0], hinge[1]), (hinge[1], hinge[0]), (hinge[0], hinge[2]), (hinge[2], hinge[0])]:
                        stick_ind[m] = 2
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)


        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T], vel[frame_T], cfg

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg
