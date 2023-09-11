import torch
import torch.nn as nn
from scipy.spatial import KDTree
import numpy as np


def apply_pointwise_reg(warped_xyz, xyz_, huber_fn, num_sdf_samples):
    dist = torch.norm(warped_xyz - xyz_, dim=-1)
    pw_loss = huber_fn(dist, delta=0.25) / num_sdf_samples
    return pw_loss


def apply_pointpair_reg(warped_xyz, xyz_, loss_lp, scene_per_split, num_sdf_samples):
    delta_xyz = warped_xyz - xyz_
    xyz_reshaped = xyz_.view((scene_per_split, -1, 3))
    delta_xyz_reshape = delta_xyz.view((scene_per_split, -1, 3))
    k = xyz_reshaped.shape[1] // 8
    lp_loss = torch.sum(loss_lp(
        xyz_reshaped[:, :k].view(scene_per_split, -1, 1, 3),
        xyz_reshaped[:, k:].view(scene_per_split, 1, -1, 3),
        delta_xyz_reshape[:, :k].view(scene_per_split, -1, 1, 3),
        delta_xyz_reshape[:, k:].view(scene_per_split, 1, -1, 3),
    )) / num_sdf_samples
    # lp_loss = torch.sum(
    #     loss_sm(xyz_, delta_xyz)
    # ) / num_sdf_samples
    return lp_loss


class LipschitzLoss(nn.Module):
    def __init__(self, k, reduction=None):
        super(LipschitzLoss, self).__init__()
        self.relu = nn.ReLU()
        self.k = k
        self.reduction = reduction

    def forward(self, x1, x2, y1, y2):
        l = self.relu(torch.norm(y1-y2, dim=-1) / (torch.norm(x1-x2, dim=-1)+1e-3) - self.k)
        # l = torch.clamp(l, 0.0, 5.0)    # avoid
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)


class HuberFunc(nn.Module):
    def __init__(self, reduction=None):
        super(HuberFunc, self).__init__()
        self.reduction = reduction

    def forward(self, x, delta):
        n = torch.abs(x)
        cond = n < delta
        l = torch.where(cond, 0.5 * n ** 2, n*delta - 0.5 * delta**2)
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)


class SoftL1Loss(nn.Module):
    def __init__(self, reduction=None):
        super(SoftL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, eps=0.0, lamb=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        ret = ret * (1 + lamb * torch.sign(target) * torch.sign(target-input))
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(ret)
        else:
            return torch.sum(ret)


class BiomechanicsLoss(nn.Module):
    def __init__(self, reduction=None):
        super(BiomechanicsLoss, self).__init__()
        self.reduction = reduction
        vp = 0.4
        Ep = 0.21
        Ci = torch.zeros(6, 6)
        Ci[0, 0] = 1 / Ep
        Ci[0, 1] = -vp / Ep
        Ci[0, 2] = -vp / Ep
        Ci[1, 0] = -vp / Ep
        Ci[1, 1] = 1 / Ep
        Ci[1, 2] = -vp / Ep
        Ci[2, 0] = -vp
        Ci[2, 1] = -vp
        Ci[2, 2] = 1 / Ep
        Ci[3, 3] = 2 * (1 + vp) / Ep
        Ci[4, 4] = 2 * (1 + vp) / Ep
        Ci[5, 5] = 2 * (1 + vp) / Ep
        self.C = Ci.inverse().cuda()

    def forward(self, coords, warped, gt_sdf):
        # warp: [N, 3], pred_sdf: [N, 1], coords: [N, 3]
        # new_coords = warped.requires_grad_(True)
        motion = coords - warped  # ED to t phase

        # select surface points
        index_list = torch.nonzero(gt_sdf < 1e-8, as_tuple=False).squeeze()
        # compute gradients
        u = motion[:, 0]
        v = motion[:, 1]
        w = motion[:, 2]

        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [warped], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=ux,uy,uz]
        grad_v = torch.autograd.grad(v, [warped], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=vx,vy,vz]
        grad_w = torch.autograd.grad(w, [warped], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=wx,wy,wz]
        # 3*3 Jac: [[ux, vx, wx],
        #           [uy, vy, wy],
        #           [uz, vz, wz]]

        et = torch.empty(warped.shape[0], 6).float().cuda()  # [N, 6]
        et[:, 0] = grad_u[:, 0]
        et[:, 1] = grad_v[:, 1]
        et[:, 2] = grad_w[:, 2]
        et[:, 3] = (grad_u[:, 1] + grad_v[:, 0]) / 2
        et[:, 4] = (grad_u[:, 2] + grad_w[:, 0]) / 2
        et[:, 5] = (grad_w[:, 1] + grad_v[:, 2]) / 2
        et = torch.index_select(et, dim=0, index=index_list)
        e = et.t()

        # W = ||et * C * e||
        W = torch.diag(torch.matmul(et, torch.matmul(self.C, e))).norm(dim=-1)
        return W/et.shape[0]


class BiomechanicsLoss_kdtree(nn.Module):
    def __init__(self, reduction=None):
        super(BiomechanicsLoss_kdtree, self).__init__()
        self.reduction = reduction
        vp = 0.4
        Ep = 0.21
        Ci = torch.zeros(6, 6)
        Ci[0, 0] = 1 / Ep
        Ci[0, 1] = -vp / Ep
        Ci[0, 2] = -vp / Ep
        Ci[1, 0] = -vp / Ep
        Ci[1, 1] = 1 / Ep
        Ci[1, 2] = -vp / Ep
        Ci[2, 0] = -vp
        Ci[2, 1] = -vp
        Ci[2, 2] = 1 / Ep
        Ci[3, 3] = 2 * (1 + vp) / Ep
        Ci[4, 4] = 2 * (1 + vp) / Ep
        Ci[5, 5] = 2 * (1 + vp) / Ep
        self.C = Ci.inverse().cuda()

    def forward(self, new_xyz, xyz, gt_sdf): # ED to t phase
        # warp: [N, 3], pred_sdf: [N, 1], coords: [N, 3]
        # new_coords = warped.requires_grad_(True)
        motion = new_xyz - xyz # ED to t phase

        # select surface points
        index_list = torch.nonzero(gt_sdf < 1e-8, as_tuple=False).squeeze()
        motion_inside = torch.index_select(motion, dim=0, index=index_list)
        warped_inside = torch.index_select(new_xyz, dim=0, index=index_list)

        import pyrender
        # points = warped.cpu().detach().numpy()
        # colors = np.zeros(points.shape)
        # cloud = pyrender.Mesh.from_points(points, colors=colors)
        # scene = pyrender.Scene()
        # scene.add(cloud)
        # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

        warped_inside_np = warped_inside.cpu().detach().numpy()
        tree = KDTree(warped_inside_np)
        distances, indices = tree.query(warped_inside_np, k=2)
        tree_list1 = torch.IntTensor(np.array(np.nonzero(distances[:, 1] > 1e-8)).squeeze()).cuda()
        tree_list2 = torch.IntTensor(indices[tree_list1.cpu(), 1].squeeze()).cuda()
        motion_inside_compute = torch.index_select(motion_inside, dim=0, index=tree_list2)
        warped_inside_compute = torch.index_select(warped_inside, dim=0, index=tree_list2)
        motion_inside = torch.index_select(motion_inside, dim=0, index=tree_list1)
        warped_inside = torch.index_select(warped_inside, dim=0, index=tree_list1)

        # # 创建一个线网格对象
        # vertices = np.vstack((warped_inside_compute[0:1000].cpu().detach().numpy(),
        #                       warped_inside[0:1000].cpu().detach().numpy()))
        # faces = np.stack((np.arange(1000), np.arange(1000, 2000)), axis=1)
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(vertices)
        # line_set = o3d.geometry.LineSet(points=pcd.points, lines=o3d.utility.Vector2iVector(faces))
        # o3d.visualization.draw_geometries([line_set])


        motion_detla = motion_inside_compute - motion_inside
        cords_delta = warped_inside_compute - warped_inside + 1e-8
        du = motion_detla[:, 0]
        dv = motion_detla[:, 1]
        dw = motion_detla[:, 2]
        dx = cords_delta[:, 0]
        dy = cords_delta[:, 1]
        dz = cords_delta[:, 2]

        grad_u = torch.stack((torch.div(du, dx), torch.div(du, dy), torch.div(du, dz)), dim=1)  # [N, 3=ux,uy,uz]
        grad_v = torch.stack((torch.div(dv, dx), torch.div(dv, dy), torch.div(dv, dz)), dim=1)  # [N, 3=vx,vy,vz]
        grad_w = torch.stack((torch.div(dw, dx), torch.div(dw, dy), torch.div(dw, dz)), dim=1)  # [N, 3=wx,wy,wz]
        # # 3*3 Jac: [[ux, vx, wx],
        # #           [uy, vy, wy],
        # #           [uz, vz, wz]]
        #
        et = torch.empty(warped_inside.shape[0], 6).float().cuda()  # [N, 6]
        et[:, 0] = grad_u[:, 0]
        et[:, 1] = grad_v[:, 1]
        et[:, 2] = grad_w[:, 2]
        et[:, 3] = (grad_u[:, 1] + grad_v[:, 0]) / 2
        et[:, 4] = (grad_u[:, 2] + grad_w[:, 0]) / 2
        et[:, 5] = (grad_w[:, 1] + grad_v[:, 2]) / 2
        e = et.t()
        # W = ||et * C * e||
        W = torch.diag(torch.matmul(et, torch.matmul(self.C, e))).norm(dim=-1)
        return W/et.shape[0]


# sdf regression loss from Sitzmannn et al. 2020
class EikonalLoss(nn.Module):
    def __init__(self, reduction=None):
        super(EikonalLoss, self).__init__()
        self.reduction = reduction

    def forward(self, coords, warped, pred_sdf, gt_sdf):
        motion = coords - warped  # ED to t

        # compute gradients
        u = motion[:, 0]
        v = motion[:, 1]
        w = motion[:, 2]
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [warped], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=ux,uy,uz]
        grad_v = torch.autograd.grad(v, [warped], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=vx,vy,vz]
        grad_w = torch.autograd.grad(w, [warped], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=wx,wy,wz]
        grad_deform = torch.stack([grad_u, grad_v, grad_w], dim=2)  # gradient of deformation wrt. input position
        # grad_temp = torch.autograd.grad(sdf, [new_coords], grad_outputs=torch.ones_like(sdf), create_graph=True)[
        #     0]  # normal direction in template space
        grad_sdf = torch.autograd.grad(pred_sdf, [warped], grad_outputs=torch.ones_like(pred_sdf), create_graph=True)[
            0]  # normal direction in original shape space


        # sdf_constraint = torch.where(gt_sdf != -1, torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5),
        #                              torch.zeros_like(pred_sdf))

        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))

        # normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(grad_sdf, gt_normals, dim=-1)[..., None],
        #                                 torch.zeros_like(grad_sdf[..., :1]))

        grad_constraint = torch.abs(grad_sdf.norm(dim=-1) - 1)

        # deformation smoothness prior
        grad_deform_constraint = grad_deform.norm(dim=-1)

        # normal consistency prior
        # grad_temp_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient_temp, gt_normals, dim=-1)[..., None],
        #                                 torch.zeros_like(gradient_temp[..., :1]))

        # latent code prior
        # embeddings_constraint = torch.mean(embeddings ** 2)

        return inter_constraint.mean() * 5e2 + \
               grad_constraint.mean() * 5e1 + \
               grad_deform_constraint.mean() * 5



if __name__ == '__main__':

    ################## bio loss test###################
    # vp = 0.4
    # Ep = 0.21
    # Ci = torch.zeros(6, 6)
    # Ci[0, 0] = 1 / Ep
    # Ci[0, 1] = -vp / Ep
    # Ci[0, 2] = -vp / Ep
    # Ci[1, 0] = -vp / Ep
    # Ci[1, 1] = 1 / Ep
    # Ci[1, 2] = -vp / Ep
    # Ci[2, 0] = -vp
    # Ci[2, 1] = -vp
    # Ci[2, 2] = 1 / Ep
    # Ci[3, 3] = 2 * (1 + vp) / Ep
    # Ci[4, 4] = 2 * (1 + vp) / Ep
    # Ci[5, 5] = 2 * (1 + vp) / Ep
    # C = Ci.inverse()
    #
    # # warp: [N, 3], pred_sdf: [N, 1], coords: [N, 3]
    # pred_sdf = torch.tensor([0, 1, 0, 1])
    # coords = torch.tensor([[1, 2, 3], [4, 5, 6], [3, 2, 1], [1, 1, 1]]).float().requires_grad_(True)
    # warp = torch.empty(coords.shape[0], 3).float()
    # warp[:, 0] = 2 *coords[:, 0]*coords[:, 0] + coords[:, 1] + coords[:, 2]
    # warp[:, 1] = coords[:, 0] * coords[:, 0] + 2 * coords[:, 1] + coords[:, 2]
    # warp[:, 2] = coords[:, 0] * coords[:, 0] + coords[:, 1] + 2 * coords[:, 2]
    # # print(warp)
    # # warp = torch.FloatTensor([[ 7.,  8.,  9.],
    # #     [43., 32., 33.],
    # #     [21., 14., 13.],
    # #     [ 4.,  4.,  4.]]).requires_grad_(True)
    #
    # print(coords)
    # # select surface points
    # index_list = torch.nonzero(pred_sdf < 1e-8, as_tuple=False).squeeze()
    # # coords = torch.index_select(coords, dim=0, index=index_list)
    # # warp = torch.index_select(warp, dim=0, index=index_list)
    #
    # u = warp[:, 0]
    # v = warp[:, 1]
    # w = warp[:, 2]
    #
    # grad_outputs = torch.ones_like(u)
    # grad_u = torch.autograd.grad(u, [coords], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=ux,uy,uz]
    # grad_v = torch.autograd.grad(v, [coords], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=vx,vy,vz]
    # grad_w = torch.autograd.grad(w, [coords], grad_outputs=grad_outputs, create_graph=True)[0]  # [N, 3=wx,wy,wz]
    # print(grad_u)
    # print(grad_v)
    # print(grad_w)
    # # 3*3 Jac: [[ux, vx, wx],
    # #           [uy, vy, wy],
    # #           [uz, vz, wz]]
    # grad_deform = torch.stack([grad_u, grad_v, grad_w], dim=2)  # gradient of deformation wrt. input position, [N, 3, 3]
    # print(grad_deform.shape)
    #
    # epsilon = torch.empty(3, 3).float()
    # et = torch.empty(coords.shape[0], 6).float()  # [N, 6]
    # et[:, 0] = grad_u[:, 0]
    # et[:, 1] = grad_v[:, 1]
    # et[:, 2] = grad_w[:, 2]
    # et[:, 3] = (grad_u[:, 1] + grad_v[:, 0]) / 2
    # et[:, 4] = (grad_u[:, 2] + grad_w[:, 0]) / 2
    # et[:, 5] = (grad_w[:, 1] + grad_v[:, 2]) / 2
    # et = torch.index_select(et, dim=0, index=index_list)
    # e = et.t()
    #
    # W = torch.diag(torch.matmul(et, torch.matmul(C, e))).norm(dim=-1)
    # print(W)


    #############################################
    # t = torch.tensor([0, 1, 1])
    # x = torch.FloatTensor([[2], [1], [3]])
    # x.requires_grad = True
    # y = x.clone()
    # index_nonED = torch.nonzero(t).squeeze().tolist()
    # x_ = x[index_nonED, :]
    # y_ = x_ * x_
    #
    # for i in range(len(index_nonED)):
    #     y[index_nonED[i]] = y_[i]
    # y.backward(torch.ones_like(x))
    # print(x.grad)
    #

    ############################################# bio use knn search
    vp = 0.4
    Ep = 0.21
    Ci = torch.zeros(6, 6)
    Ci[0, 0] = 1 / Ep
    Ci[0, 1] = -vp / Ep
    Ci[0, 2] = -vp / Ep
    Ci[1, 0] = -vp / Ep
    Ci[1, 1] = 1 / Ep
    Ci[1, 2] = -vp / Ep
    Ci[2, 0] = -vp
    Ci[2, 1] = -vp
    Ci[2, 2] = 1 / Ep
    Ci[3, 3] = 2 * (1 + vp) / Ep
    Ci[4, 4] = 2 * (1 + vp) / Ep
    Ci[5, 5] = 2 * (1 + vp) / Ep
    C = Ci.inverse()

    # warp: [N, 3], pred_sdf: [N, 1], coords: [N, 3]
    gt_sdf = torch.tensor([0, -1, 0, -1])
    warped = torch.tensor([[1, 2, 3], [1.0001, 2.001, 3.01], [1, 2, 3], [1.01, 2.0001, 3.001]]).float().requires_grad_(True)
    motion = torch.empty(warped.shape[0], 3).float()
    motion[:, 0] = 2 *warped[:, 0]*warped[:, 0] + warped[:, 1] + warped[:, 2]
    motion[:, 1] = warped[:, 0] * warped[:, 0] + 2 * warped[:, 1] + warped[:, 2]
    motion[:, 2] = warped[:, 0] * warped[:, 0] + warped[:, 1] + 2 * warped[:, 2]
    print(motion)

    index_list = torch.nonzero(gt_sdf < 1e-8, as_tuple=False).squeeze()
    motion_inside = torch.index_select(motion, dim=0, index=index_list)
    warped_inside = torch.index_select(warped, dim=0, index=index_list)
    warped_inside_np = warped_inside.detach().numpy()
    tree = KDTree(warped_inside_np)
    distances, indices = tree.query(warped_inside_np, k=2)
    tree_list1 = np.array(np.nonzero(distances[:, 1] > 1e-8)).squeeze()
    tree_list2 = indices[tree_list1, 1].squeeze()
    motion_inside_compute = torch.index_select(motion_inside, dim=0, index=torch.IntTensor(tree_list2))
    warped_inside_compute = torch.index_select(warped_inside, dim=0, index=torch.IntTensor(tree_list2))
    motion_inside = torch.index_select(motion_inside, dim=0, index=torch.IntTensor(tree_list1))
    warped_inside = torch.index_select(warped_inside, dim=0, index=torch.IntTensor(tree_list1))

    motion_detla = motion_inside_compute - motion_inside
    cords_delta = warped_inside_compute - warped_inside + 1e-8
    du = motion_detla[:, 0]
    dv = motion_detla[:, 1]
    dw = motion_detla[:, 2]
    dx = cords_delta[:, 0]
    dy = cords_delta[:, 1]
    dz = cords_delta[:, 2]

    print(du)
    print(dx)
    ux = torch.div(du, dx)
    print(ux)
    grad_u = torch.stack((torch.div(du, dx), torch.div(du, dy), torch.div(du, dz)), dim=1)  # [N, 3=ux,uy,uz]
    grad_v = torch.stack((torch.div(dv, dx), torch.div(dv, dy), torch.div(dv, dz)), dim=1)  # [N, 3=vx,vy,vz]
    grad_w = torch.stack((torch.div(dw, dx), torch.div(dw, dy), torch.div(dw, dz)), dim=1)  # [N, 3=wx,wy,wz]
    # # 3*3 Jac: [[ux, vx, wx],
    # #           [uy, vy, wy],
    # #           [uz, vz, wz]]
    #
    epsilon = torch.empty(3, 3).float()
    et = torch.empty(warped_inside.shape[0], 6).float() # [N, 6]
    et[:, 0] = grad_u[:, 0]
    et[:, 1] = grad_v[:, 1]
    et[:, 2] = grad_w[:, 2]
    et[:, 3] = (grad_u[:, 1] + grad_v[:, 0]) / 2
    et[:, 4] = (grad_u[:, 2] + grad_w[:, 0]) / 2
    et[:, 5] = (grad_w[:, 1] + grad_v[:, 2]) / 2
    e = et.t()
    W = torch.diag(torch.matmul(et, torch.matmul(C, e))).norm(dim=-1)
    print(W)
    #


    ################## test pp-loss
    # warped_xyz = torch.tensor([[1, 2, 3], [4, 5, 6], [3, 2, 1], [1, 1, 1], [2,3,4], [3,1,5]])
    # xyz_ = torch.tensor([[5,2,4], [3,5,2], [6,4,2], [2,5,2], [4, 5, 6], [3, 2, 1]])
    # delta_xyz = warped_xyz - xyz_
    # xyz_reshaped = xyz_.view((1, -1, 3))
    # delta_xyz_reshape = delta_xyz.view((1, -1, 3))
    # k = xyz_reshaped.shape[1] // 3
    # a = xyz_reshaped[:, :k].view(1, -1, 1, 3)
    # b = xyz_reshaped[:, k:].view(1, 1, -1, 3)
    # c = a-b
    # print(a)
    # print(b)
    # print(c)
