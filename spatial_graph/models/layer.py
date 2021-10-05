from torch import nn
import torch
import torch.nn.functional as F


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


# add our model here
class E_GCL_vel_mechanics(nn.Module):

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL_vel_mechanics, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        n_basis_stick = 1
        n_basis_hinge = 3

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.coord_mlp_w_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.center_mlp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, input_nf))

        self.f_stick_mlp = nn.Sequential(
            nn.Linear(n_basis_stick * n_basis_stick, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_stick)
        )
        self.f_hinge_mlp = nn.Sequential(
            nn.Linear(n_basis_hinge * n_basis_hinge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_hinge)
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, others=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        if others is not None:  # can concat h here
            agg = torch.cat([others, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        f = agg * self.coords_weight
        return coord + f, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def update(self, x, v, f, h, node_index, type='Isolated'):
        if type == 'Isolated':
            _x, _v, _f, _h = x[node_index], v[node_index], f[node_index], h[node_index]
            _a = _f / 1.
            _v = self.coord_mlp_vel(_h) * _v + _a
            _x = _x + _v
            # put the updated x, v (local object) back to x, v (global graph)
            x[node_index] = _x
            v[node_index] = _v
            return x, v

        elif type == 'Stick':
            id1, id2 = node_index[..., 0], node_index[..., 1]
            _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
            _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]
            _x0, _v0, _f0 = (_x1 + _x2) / 2, (_v1 + _v2) / 2, _f1 + _f2

            def apply_f(cur_x, cur_v, cur_f):
                _X = torch.stack((cur_f,), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.f_stick_mlp(_invariant_X)
                message = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                return message

            messages = [apply_f(_x1, _v1, _f1), apply_f(_x2, _v2, _f2)]
            _a0 = sum(messages) / len(messages)

            J = torch.sum((_x1 - _x0) ** 2, dim=-1, keepdim=True) + torch.sum((_x2 - _x0) ** 2, dim=-1, keepdim=True)
            _beta1, _beta2 = torch.cross((_x1 - _x0), _f1) / J, torch.cross((_x2 - _x0), _f2) / J

            _beta = _beta1 + _beta2  # sum pooling over local object  # [B*N', 3]

            # compute c metrics
            _r, _v = (_x1 - _x2) / 2, (_v1 - _v2) / 2
            _w = torch.cross(F.normalize(_r, dim=-1, p=2), _v) / torch.norm(_r, dim=-1, p=2, keepdim=True).clamp_min(
                1e-5)  # [B*N', 3]

            trans_h1, trans_h2 = self.center_mlp(_h1), self.center_mlp(_h2)
            _h_c = trans_h1 + trans_h2

            _w = self.coord_mlp_w_vel(_h_c) * _w + _beta  # [B*N', 3]
            _v0 = self.coord_mlp_vel(_h_c) * _v0 + _a0  # [B*N', 3]
            _x0 = _x0 + _v0
            _theta = torch.norm(_w, p=2, dim=-1)  # [B*N']
            rot = self.compute_rotation_matrix(_theta, F.normalize(_w, p=2, dim=-1))

            _r = torch.einsum('bij,bjk->bik', rot, _r.unsqueeze(-1)).squeeze(-1)  # [B*N', 3]
            _x1 = _x0 + _r
            _x2 = _x0 - _r
            _v1 = _v0 + torch.cross(_w, _r)
            _v2 = _v0 + torch.cross(_w, - _r)

            # put the updated x, v (local object) back to x, v (global graph)
            x[id1], x[id2] = _x1, _x2
            v[id1], v[id2] = _v1, _v2
            return x, v

        elif type == 'Hinge':
            id0, id1, id2 = node_index[..., 0], node_index[..., 1], node_index[..., 2]
            _x0, _v0, _f0, _h0 = x[id0], v[id0], f[id0], h[id0]
            _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
            _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]

            def apply_f(cur_x, cur_v, cur_f):
                _X = torch.stack((cur_f, cur_x - _x0, cur_v - _v0), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.f_hinge_mlp(_invariant_X)
                message = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                return message

            messages = [apply_f(_x0, _v0, _f0), apply_f(_x1, _v1, _f1), apply_f(_x2, _v2, _f2)]
            _a0 = sum(messages) / len(messages)

            def apply_g(cur_x, cur_f):
                message = torch.cross(cur_x - _x0, cur_f - _a0) / torch.sum((cur_x - _x0) ** 2, dim=-1, keepdim=True)
                return message

            _beta1, _beta2 = apply_g(_x1, _f1), apply_g(_x2, _f2)

            def compute_c_metrics(cur_x, cur_v):
                cur_r, relative_v = cur_x - _x0, cur_v - _v0
                cur_w = torch.cross(F.normalize(cur_r, dim=-1, p=2), relative_v) / torch.norm(
                    cur_r, dim=-1, p=2, keepdim=True).clamp_min(1e-5)
                return cur_r, cur_w

            _r1, _w1 = compute_c_metrics(_x1, _v1)
            _r2, _w2 = compute_c_metrics(_x2, _v2)

            trans_h1, trans_h2 = self.center_mlp(_h1), self.center_mlp(_h2)
            _h_c = trans_h1 + trans_h2
            _v0 = self.coord_mlp_vel(_h_c) * _v0 + _a0  # [B*N', 3]
            _x0 = _x0 + _v0

            def update_c_metrics(rot_func, cur_w, cur_beta, cur_r, cur_h):
                cur_w = self.coord_mlp_w_vel(cur_h) * cur_w + cur_beta  # [B*N', 3]
                cur_theta = torch.norm(cur_w, p=2, dim=-1)  # [B*N']
                cur_rot = rot_func(cur_theta, F.normalize(cur_w, p=2, dim=-1))
                cur_r = torch.einsum('bij,bjk->bik', cur_rot, cur_r.unsqueeze(-1)).squeeze(-1)  # [B*N', 3]
                return cur_r, cur_w

            _r1, _w1 = update_c_metrics(self.compute_rotation_matrix, _w1, _beta1, _r1, _h1)
            _r2, _w2 = update_c_metrics(self.compute_rotation_matrix, _w2, _beta2, _r2, _h2)

            _x1, _x2 = _x0 + _r1, _x0 + _r2
            _v1, _v2 = _v0 + torch.cross(_w1, _r1), _v0 + torch.cross(_w2, _r2)

            # put the updated x, v (local object) back to x, v (global graph)
            x[id0], x[id1], x[id2] = _x0, _x1, _x2
            v[id0], v[id1], v[id2] = _v0, _v1, _v2

            return x, v
        else:
            raise NotImplementedError('Unknown object type:', type)

    def forward(self, h, edge_index, x, v, cfg, edge_attr=None, node_attr=None):
        """
        :param h:
        :param edge_index:
        :param x:
        :param v:
        :param cfg: {'isolated': idx, 'stick': [c0, c1], 'hinge': [c0, c1, c2]}
        :param edge_attr:
        :param node_attr:
        :return:
        """

        # aggregate force (equivariant message passing on the whole graph)
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, x)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [B*M, Ef], the global invariant message
        _, f = self.coord_model(x, edge_index, coord_diff, edge_feat)  # [B*N, 3]

        for type in cfg:
            x, v = self.update(x, v, f, h, node_index=cfg[type], type=type)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, others=h)

        return h, x, v, edge_attr

    @staticmethod
    def compute_rotation_matrix(theta, d):
        x, y, z = torch.unbind(d, dim=-1)
        cos, sin = torch.cos(theta), torch.sin(theta)
        ret = torch.stack((
            cos + (1 - cos) * x * x,
            (1 - cos) * x * y - sin * z,
            (1 - cos) * x * z + sin * y,
            (1 - cos) * x * y + sin * z,
            cos + (1 - cos) * y * y,
            (1 - cos) * y * z - sin * x,
            (1 - cos) * x * z - sin * y,
            (1 - cos) * y * z + sin * x,
            cos + (1 - cos) * z * z,
        ), dim=-1)

        return ret.reshape(-1, 3, 3)  # [B*N, 3, 3]