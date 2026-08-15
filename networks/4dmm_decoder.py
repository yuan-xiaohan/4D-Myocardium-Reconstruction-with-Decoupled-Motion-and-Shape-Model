"""The cardiac_4d decoder, checkpoint-compatible with the trained ACDC models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeNet(nn.Module):
    def __init__(self, latent_size, dims, dropout=None, dropout_prob=0.0,
                 norm_layers=(), latent_in=(), weight_norm=False,
                 xyz_in_all=None, use_tanh=False, latent_dropout=False):
        super().__init__()
        dims = [latent_size + 3] + dims + [1]
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_tanh = use_tanh
        self.dropout_prob = dropout_prob
        self.dropout = dropout

        for layer in range(self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3
            linear = nn.Linear(dims[layer], out_dim)
            if weight_norm and layer in norm_layers:
                linear = nn.utils.weight_norm(linear)
            setattr(self, "lin" + str(layer), linear)
            if not weight_norm and norm_layers is not None and layer in norm_layers:
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

    def forward(self, inputs):
        xyz = inputs[:, -3:]
        if inputs.shape[1] > 3 and self.latent_dropout:
            x = torch.cat([F.dropout(inputs[:, :-3], p=0.2, training=self.training), xyz], 1)
        else:
            x = inputs

        for layer in range(self.num_layers - 1):
            if layer in self.latent_in:
                x = torch.cat([x, inputs], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = getattr(self, "lin" + str(layer))(x)
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if self.norm_layers is not None and layer in self.norm_layers and not self.weight_norm:
                    x = getattr(self, "bn" + str(layer))(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return self.th(x)


class MotionNet(nn.Module):
    def __init__(self, dim=4, in_features=256, out_features=3, num_filters=32,
                 activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.num_filters = num_filters
        self.activ = activation
        self.fc0 = nn.Linear(self.dimz, num_filters * 16)
        self.fc1 = nn.Linear(self.dimz + num_filters * 16, num_filters * 8)
        self.fc2 = nn.Linear(self.dimz + num_filters * 8, num_filters * 4)
        self.fc3 = nn.Linear(self.dimz + num_filters * 4, num_filters * 2)
        self.fc4 = nn.Linear(self.dimz + num_filters * 2, num_filters)
        self.fc5 = nn.Linear(num_filters, out_features)
        # Retain the original attribute layout; ModuleList would alter checkpoint keys.
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    def forward(self, x):
        x_out = x
        for dense in self.fc[:4]:
            x_out = self.activ(dense(x_out))
            x_out = torch.cat([x_out, x], dim=-1)
        return self.fc5(self.activ(self.fc4(x_out)))


class Decoder(nn.Module):
    def __init__(self, motionnet_kargs, shapenet_kargs):
        super().__init__()
        self.motion_net = MotionNet(**motionnet_kargs)
        self.shape_net = ShapeNet(**shapenet_kargs)

    def decode(self, x, c_s):
        batch_size, point_count, _ = x.shape
        x_flat = x.reshape(-1, 3)
        shape_codes = c_s[:, None, :].expand(batch_size, point_count, -1).reshape(-1, c_s.shape[1])
        return self.shape_net(torch.cat([shape_codes, x_flat], dim=-1))

    def deform_points(self, x, t, c_m):
        batch_size, point_count, _ = x.shape
        new_coords = x.clone()
        non_ed = torch.nonzero(t != 0, as_tuple=True)[0]
        if non_ed.numel() == 0:
            return new_coords
        selected = x.index_select(0, non_ed)
        t_selected = t.index_select(0, non_ed).reshape(-1, 1, 1).expand(-1, point_count, 1)
        cm_selected = c_m.index_select(0, non_ed)[:, None, :].expand(-1, point_count, -1)
        features = torch.cat([selected, t_selected, cm_selected], dim=-1).reshape(-1, 4 + c_m.shape[1])
        warped = selected + self.motion_net(features).reshape(-1, point_count, 3)
        # Keep the original assignment order for numerical training parity while
        # retaining robust tensor-shaped indices and the all-ED early return.
        for index, batch_index in enumerate(non_ed.tolist()):
            new_coords[batch_index, :, :] = warped[index, :, :]
        return new_coords

    def forward(self, coords, t, c_m, c_s):
        new_coords = self.deform_points(coords, t, c_m)
        return new_coords, self.decode(new_coords, c_s)
