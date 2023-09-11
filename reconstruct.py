import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import deep_sdf
import deep_sdf.workspace as ws
import deep_sdf.loss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def reconstruct(
    decoder,
    num_iterations,
    npz_filename,
    frame_num,
    c_s,
    c_m,
    clamp_dist,
    N=1000,
    lr=5e-4,
    l2reg=False,
):

    sdf_error = 0.0
    loss_l1 = torch.nn.L1Loss()
    loss_lp = torch.nn.DataParallel(deep_sdf.loss.LipschitzLoss(k=0.5, reduction="sum"))
    huber_fn = deep_sdf.loss.HuberFunc(reduction="sum")

    for epoch in range(num_iterations):
        sdf_data = dict()
        phase_list = os.listdir(npz_filename)
        samples = []
        ts = []
        for frame in phase_list:
            frame_filename = os.path.join(npz_filename, frame)
            sample, t = deep_sdf.dataset.get_sdf_samples_test(frame_filename, N)
            samples.append(sample)
            ts.append(t)
        samples = torch.stack(samples)
        ts = torch.from_numpy(np.array(ts))
        sdf_data["p_sdf"] = samples  # [frame_num, N, 4]
        sdf_data["t"] = ts  # [frame_num, 1]

        decoder.eval()
        adjust_learning_rate(lr, optimizer, epoch, decreased_by, adjust_lr_every)
        optimizer.zero_grad()

        total_loss = []
        p_sdf = sdf_data["p_sdf"].reshape(-1, 4)  # [b*frame_num*N, 4]
        t = sdf_data["t"].view(-1).unsqueeze(-1).repeat(1, N).view(-1).cuda()  # [b*frame_num*N, ]
        num_sdf_samples = p_sdf.shape[0]

        xyz = p_sdf[:, 0:3].view(-1, 3).cuda()  # [b*frame_num*N, 3]
        sdf_gt = p_sdf[:, 3]  # [b, 1, frame_num, N, 1]
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist).cuda()

        indices = torch.LongTensor([0, ])
        #### seq_index: [b, ]->[b, 1]->[b, frame_num*N]->[b*frame_num*N, ]
        seq_index = indices.unsqueeze(-1).repeat(1, frame_num * N).view(-1)
        #### frame_index: [frame_num, ]->[frame_num, 1]->[frame_num, N]
        # ->[frame_num*N, ]->[frame_num*N*b, ]
        frame_index = torch.IntTensor(list(range(frame_num))).unsqueeze(-1).repeat(1, N)\
            .view(-1).repeat(indices.shape[0])
        m_index = seq_index * frame_num + frame_index

        cs_vecs = c_s[seq_index].cuda()  # [b*frame_num*N, Cs_size]
        cm_vecs = c_m[m_index].cuda()  #

        new_xyz, sdf_pred = decoder(xyz, t, cm_vecs, cs_vecs)

        if epoch == 0:
            new_xyz, sdf_pred = decoder(xyz, t, cm_vecs, cs_vecs)

        sdf_pred = torch.clamp(sdf_pred.squeeze(1), -clamp_dist, clamp_dist)

        sdf_loss = loss_l1(sdf_pred, sdf_gt)/num_sdf_samples
        loss = sdf_loss
        if l2reg:
            loss += 1e-4 * torch.mean(cs_vecs.pow(2))
            index_nonED = torch.nonzero(t).squeeze()
            loss += 1e-4 * torch.mean(cm_vecs[index_nonED, :].pow(2))

        # pointwise loss
        use_pointwise_loss = False
        if use_pointwise_loss:
            pw_loss = deep_sdf.loss.apply_pointwise_reg(new_xyz, xyz, huber_fn, num_sdf_samples)
            loss += 1e-4 * pw_loss.cuda()

        # pointpair loss
        use_pointpair_loss = False
        if use_pointpair_loss:
            lp_loss = deep_sdf.loss.apply_pointpair_reg(new_xyz, xyz, loss_lp, 1, num_sdf_samples)
            loss += 1e-4 * lp_loss.cuda()

        loss.backward()
        optimizer.step()
        sdf_error = sdf_loss.item()
        print("epoch {}, sdf_loss = {:.9e}".format(epoch, sdf_error))
        # if epoch % 50 == 0:
        #     print("epoch {}, sdf_loss = {:.6f}".format(epoch, sdf_error))

    return sdf_error, c_s, c_m


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=50,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=10,
        help="random seed",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=128,
        help="Marching cube resolution.",
    )

    lr = 5e-3
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    deep_sdf.configure_logging(args)
    args.resolution = 128
    max_batch = int(2 ** 17)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    frame_num = specs["FrameNum"]
    ini_path = specs["IniPath"]
    Cs_size = specs["CsLength"]
    Cm_size = specs["CmLength"]
    N = specs["SamplesPerScene"]  # N must be even number

    decoder = arch.Decoder(**specs["NetworkSpecs"]).cuda()

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    seqfiles = deep_sdf.dataset.get_instance_filenames(args.data_source, split)

    logging.debug(decoder)

    err_sum = 0.0
    save_latvec_only = False

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    reconstruction_motion_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_motions_subdir
    )
    if not os.path.isdir(reconstruction_motion_dir):
        os.makedirs(reconstruction_motion_dir)

    clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    #### for test set
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(args.iterations)/2

    for ii, npz in enumerate(seqfiles):
        phase_list = os.listdir(npz)
        frame_num = len(phase_list)
        c_s = torch.ones(1, Cs_size).normal_(mean=0, std=0.1).cuda()  # [1, Cs_size]
        c_m = torch.ones(frame_num, Cm_size).normal_(mean=0, std=1.0 / np.sqrt(Cm_size)).cuda()  # [frame_num, Cm_size]

        c_s.requires_grad = True  # [1, Cs_size]
        c_m.requires_grad = True  # [frame_num, Cm_size]

        optimizer = torch.optim.Adam([c_s, c_m], lr=5e-4)

        start = time.time()
        err, c_s, c_m = reconstruct(
            decoder,
            int(args.iterations),
            npz,
            frame_num,
            c_s,
            c_m,
            0.1,
            N=N,
            lr=lr,
            l2reg=True
        )
        logging.info("reconstruct time: {}".format(time.time() - start))
        logging.info("reconstruction error: {}".format(err))
        err_sum += err

        for phase_idx in range(len(phase_list)):
            full_filename = os.path.join(npz, phase_list[phase_idx])
            file_name = os.path.split(npz)[1] + "_" + os.path.splitext(phase_list[phase_idx])[0]
            print("reconstruct: " + file_name)
            data = np.load(full_filename)
            Ti = data["Ti"]
            offset = data["offset"]
            scale = data["scale"]
            mesh_filename = os.path.join(reconstruction_meshes_dir, file_name)
            latent_filename = os.path.join(reconstruction_codes_dir, file_name)
            motion_filename = os.path.join(reconstruction_motion_dir, file_name)
            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            c_s_vec = c_s  # [1, Cs_size]
            c_m_vecs = c_m[phase_idx, :].unsqueeze(0)  # [1, Cm_size]
            phase = torch.FloatTensor([phase_idx/(frame_num-1)]).unsqueeze(0)
            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh_4dsdf(
                        decoder, c_s_vec, c_m_vecs, phase, mesh_filename, motion_filename,
                        N=args.resolution, max_batch=max_batch, offset=offset, scale=scale, Ti=Ti)
                logging.debug("total time: {}".format(time.time() - start))

        torch.save(c_s.detach().cpu(), os.path.join(reconstruction_codes_dir, os.path.split(npz)[1] + "_cs.pth"))
        torch.save(c_m.detach().cpu(), os.path.join(reconstruction_codes_dir, os.path.split(npz)[1] + "_cm.pth"))