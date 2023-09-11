import logging
import numpy as np
import os
import torch
import torch.utils.data


def get_instance_filenames(data_source, split):
    seqfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for seq_name in split[dataset][class_name]:
                seq_filename = os.path.join(data_source, dataset, class_name, seq_name)
                seqfiles += [seq_filename]
    return seqfiles


def get_cs(data_source, split):
    cs = []
    for dataset in split:
        for class_name in split[dataset]:
            for seq_name in split[dataset][class_name]:
                cs_filename = os.path.join(data_source, seq_name+"_00.pth")
                latent = torch.load(cs_filename).squeeze()
                cs.append(latent)
    cs = torch.stack(cs)
    return cs


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def get_sdf_samples_test_pcd(filename, subsample=5000):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pcd"])
    if subsample is None:
        sample = pos_tensor  # [subsample, 4]
    else:
        random_index = (torch.rand(subsample) * pos_tensor.shape[0]).long()
        sample = torch.index_select(pos_tensor, 0, random_index).float()
    return sample


def get_sdf_samples(filename, subsample=None):
    # Make positive and negative SDF equal
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()  # random select half number of indexes
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    sample = torch.cat([sample_pos, sample_neg], 0)
    randidx = torch.randperm(sample.shape[0])
    sample = torch.index_select(sample, 0, randidx)  # [subsample, 4]
    t = npz["t"]

    return sample, t


def get_sdf_samples_test(filename, subsample=None):
    npz = np.load(filename)
    t = npz["t"]
    pos_tensor = torch.from_numpy(npz["pcd"])
    if subsample is None:
        sample = pos_tensor  # [subsample, 4]
    else:
        random_index = (torch.rand(subsample) * pos_tensor.shape[0]).long()
        sample = torch.index_select(pos_tensor, 0, random_index).float()
    return sample, t


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.seqfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.seqfiles))
            + " sequences from data source "
            + data_source
        )

    def __len__(self):
        return len(self.seqfiles)

    def __getitem__(self, idx):
        data = dict()
        frame_list = os.listdir(self.seqfiles[idx])
        samples = []
        ts = []
        for frame in frame_list:
            frame_filename = os.path.join(self.seqfiles[idx], frame)
            sample, t = get_sdf_samples(frame_filename, self.subsample)
            samples.append(sample)
            ts.append(t)
        samples = torch.stack(samples)
        ts = torch.from_numpy(np.array(ts))
        data["p_sdf"] = samples  # [frame_num, N, 4]
        data["t"] = ts  # [frame_num, 1]
        return data, idx


class SDFSamples_ED(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.seqfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.seqfiles))
            + " sequences from data source "
            + data_source
        )

    def __len__(self):
        return len(self.seqfiles)

    def __getitem__(self, idx):
        data = dict()
        frame_list = os.listdir(self.seqfiles[idx])
        samples = []
        ts = []
        for frame in frame_list:
            frame_filename = os.path.join(self.seqfiles[idx], frame)
            sample, t = get_sdf_samples(frame_filename, self.subsample)
            samples.append(sample)
            ts.append(t)
        samples = torch.stack(samples)
        ts = torch.from_numpy(np.array(ts))
        data["p_sdf"] = samples.float()  # [frame_num, N, 4]
        data["t"] = ts.float()  # [frame_num, 1]
        return data, idx



if __name__ == '__main__':
    ### get .json
    data_dir = r"\\SEUVCL-DATA-03\Data03Training\0518_4dsdf_yxh\data_acdc\LVV\Processed\test"
    instance_list = os.listdir(data_dir)
    for instance in instance_list:
        print('     "{}",'.format(os.path.splitext(instance)[0]))


    # import json
    # import torch.utils.data as data_utils
    # data_source = r"\\SEUVCL-DATA-03\Data03Training\0518_4dsdf_yxh\data_seq"
    # train_split_file = r"../examples/mini/train.json"
    # with open(train_split_file, "r") as f:
    #     train_split = json.load(f)
    # num_samp_per_scene = 5000
    # sdf_dataset = SDFSamples(data_source, train_split, num_samp_per_scene)
    # frames, indices = sdf_dataset.__getitem__(0)
    #
    # sdf_loader = data_utils.DataLoader(
    #     sdf_dataset,
    #     batch_size=3,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    # )
    # for bi, (sdf_data, indices) in enumerate(sdf_loader):
    #     #### data["p_sdf"]: [b, frame_num, N, 4], data["t"]: [b, frame_num], indices: [b, ]
    #     print(bi)
    #
    # t = torch.FloatTensor([[0, 1, 2], [0, 1, 2]])  #[2, 3]
    # cs = torch.Tensor([[0, 1], [6, 7]])  #[2, 2]
    # cm = torch.Tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])  #[2*3, 2]
    #
    # seq_num = 2
    # frame_num = 3
    # N = 4
    # print(cs.shape)
    # indices = torch.IntTensor([1, 0])
    # b = indices.shape[0]  # 2
    # print(indices.unsqueeze(-1))
    # print(indices.shape)
    # a = indices.unsqueeze(-1).repeat(1, frame_num * N)
    # print(a.shape)
    # print(a)
    # B = indices.unsqueeze(-1).repeat(1, frame_num * N).view(-1)
    # B_m = indices.unsqueeze(-1).repeat(1, frame_num * N).view(-1)*frame_num
    # print(B.shape)
    # print(B_m)
    #
    # c = torch.IntTensor(list(range(frame_num))).unsqueeze(-1).repeat(1, N).view(-1).repeat(b)
    # index_m = B_m + c
    # print(c)
    # # print(indices.unsqueeze(-1).repeat(1, frame_num * N).view(-1))
    # # print(indices.unsqueeze(-1).repeat(1, frame_num * N).view(-1))
    # # final_cm = cm[B.tolist(), c.tolist(), :]
    # final_cm = cm[index_m.tolist(), :]
    # print("final_cm", final_cm.shape)
    # print(final_cm)
    #
    # final_cs = cs[B.tolist(), :]
    # print(final_cs.shape)
    # print(final_cs)
    #
    # t = t.view(-1).unsqueeze(-1).repeat(1, N).view(-1)
    # print("t_ini, ", t)
    #
    # index_ED = torch.nonzero(t == 0)
    # t_ED = t[index_ED]
    # c_m_ED = final_cm[index_ED, :]
    #
    # index_t = torch.nonzero(t)
    # t_ = t[index_t]
    # c_m_ = final_cm[index_t, :]
    #
    # t_ = t_*10
    # c_m_ = c_m_*10
    #
    # for i in range(index_t.shape[0]):
    #     t[index_t[i]] = t_[i]
    #     final_cm[index_t[i], :] = c_m_[i, :]
    # print("t_new", t)
    # print("final_new", final_cm)
    #
