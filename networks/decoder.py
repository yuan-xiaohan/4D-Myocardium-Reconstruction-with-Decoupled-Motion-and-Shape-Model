import torch.nn as nn
import torch
import torch.nn.functional as F


class ShapeModel(nn.Module):
    def __init__(
            self,
            latent_size,
            dims,
            dropout=None,
            dropout_prob=0.0,
            norm_layers=(),
            latent_in=(),
            weight_norm=False,
            xyz_in_all=None,
            use_tanh=False,
            latent_dropout=False,
    ):
        super(ShapeModel, self).__init__()

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


class MotionModel(nn.Module):
    def __init__(self,
                 dim=4,
                 in_features=256,
                 out_features=3,
                 num_filters=32,
                 activation=nn.LeakyReLU(0.2)):
        """
        Args:
          dim: dimension of input points.
          in_features: length of latent code.
          out_features: Length of output features.
          num_filters: Width of the second to last layer.
          activation: activation function.
        """
        super(MotionModel, self).__init__()
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
        self.fc4 = nn.Linear(self.dimz + num_filters * 2, num_filters * 1)
        self.fc5 = nn.Linear(num_filters * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    def forward(self, x):
        x_ = x
        for dense in self.fc[:4]:
            x_ = self.activ(dense(x_))
            x_ = torch.cat([x_, x], dim=-1)
        x_ = self.activ(self.fc4(x_))
        x_ = self.fc5(x_)
        return x_


class Decoder(nn.Module):
    def __init__(self, motionmodel_kargs, shapemodel_kargs):
        super(Decoder, self).__init__()
        self.motion_net = MotionModel(**motionmodel_kargs)
        self.shape_net = ShapeModel(**shapemodel_kargs)

    def Deformation(self, x, t, c_m):
        coords_ED = x.clone()
        index_nonED = torch.nonzero(t).squeeze().tolist()
        x_ = x[index_nonED, :]
        t_ = t[index_nonED].unsqueeze(-1)
        c_m_ = c_m[index_nonED, :]

        deform_feat = torch.cat([x_, t_, c_m_], dim=-1).float()  # [b*frame_num*N, 3+1+Cm_size]
        offset = self.motion_net(deform_feat)
        x_ = x_ + offset

        for i in range(len(index_nonED)):
            coords_ED[index_nonED[i]] = x_[i]
        return coords_ED

    def EDShapeModel(self, x, c_s):
        decode_feat = torch.cat([c_s, x], dim=-1)  # [b*frame_num*N, Cs_size+3]
        logits = self.shape_net(decode_feat)
        return logits

    def forward(self, coords, t, c_m, c_s):
        coords_ED = self.Deformation(coords, t, c_m)
        sdf = self.EDShapeModel(coords_ED, c_s)
        return coords_ED, sdf
