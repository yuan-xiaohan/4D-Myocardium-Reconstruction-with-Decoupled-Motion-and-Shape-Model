import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn
import signal
import sys
import os
import logging
import math
import json
import tqdm
import random
import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.lr_schedule import get_learning_rate_schedules
import deep_sdf.loss as loss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, data_source, continue_from):

    logging.info("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    # data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.info(specs["NetworkSpecs"])

    Cs_size = specs["CsLength"]
    Cm_size = specs["CmLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        ws.save_model(experiment_directory, "latest.pth", decoder, epoch)
        ws.save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        ws.save_latent_vectors(experiment_directory, "latest_cm.pth", c_m, epoch)
        ws.save_latent_vectors(experiment_directory, "latest_cs.pth", c_s, epoch)

    def save_checkpoints(epoch):
        ws.save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        ws.save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        ws.save_latent_vectors(experiment_directory, str(epoch) + "_cm.pth", c_m, epoch)
        if initialize:
            torch.save(
                {"epoch": epoch, "latent_codes": c_s.detach().cpu()},
                os.path.join(experiment_directory, ws.latent_codes_subdir, str(epoch) + "_cs.pth"),
            )
        else:
            ws.save_latent_vectors(experiment_directory, str(epoch) + "_cs.pth", c_s, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    N = specs["SamplesPerScene"]  # N must be even number
    batch_size = specs["BatchSize"]
    frame_num = specs["FrameNum"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True
    save_results = specs["Reconstruct_training"]
    initialize = specs["Initialize"]
    ini_path = specs["IniPath"]
    initialize_cs = False

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(**specs["NetworkSpecs"]).cuda()

    if initialize:
        # If using a pre-trained ED shape model
        saved_model_state = torch.load(os.path.join(ini_path, "ini.pth"))
        decoder.shape_net.load_state_dict(saved_model_state["model_state_dict"])

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    #     decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.dataset.SDFSamples(data_source, train_split, N)

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    seq_num = len(sdf_dataset)

    logging.info("There are {} sequences".format(seq_num))

    logging.info(decoder)

    # initialize latent Cs for each seq and Cm for each shape
    if initialize_cs:
        Cs_path = os.path.join(ini_path, "C_s")
        c_s = torch.FloatTensor(deep_sdf.dataset.get_cs(Cs_path, train_split).detach().cpu())  # [seq_num, Cs_size]
        c_s.requires_grad = True
        c_m = torch.nn.Embedding(seq_num*frame_num, Cm_size, max_norm=code_bound)  # Embedding: [seq_num, frame_num, Cm_size]
        torch.nn.init.normal_(
            c_m.weight.data,
            0.0,
            get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(Cm_size),
        )
        c_m.requires_grad = True

        optimizer_all = torch.optim.Adam(
            [
                {
                    "params": decoder.motion_net.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": decoder.shape_net.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
                {
                    "params": c_s,
                    "lr": lr_schedules[2].get_learning_rate(0),
                },
                {
                    "params": c_m.parameters(),
                    "lr": lr_schedules[3].get_learning_rate(0),
                }
            ]
        )

    else:
        c_s = torch.nn.Embedding(seq_num, Cs_size, max_norm=code_bound)  # Embedding: [seq_num, Cs_size]
        torch.nn.init.normal_(
            c_s.weight.data,
            0.0,
            get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(Cs_size),
        )

        c_m = torch.nn.Embedding(seq_num*frame_num, Cm_size, max_norm=code_bound)  # Embedding: [seq_num, frame_num, Cm_size]
        torch.nn.init.normal_(
            c_m.weight.data,
            0.0,
            get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(Cm_size),
        )

        optimizer_all = torch.optim.Adam(
            [
                {
                    "params": decoder.motion_net.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": decoder.shape_net.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
                {
                    "params": c_s.parameters(),
                    "lr": lr_schedules[2].get_learning_rate(0),
                },
                {
                    "params": c_m.parameters(),
                    "lr": lr_schedules[3].get_learning_rate(0),
                }
            ]
        )

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_lp = torch.nn.DataParallel(loss.LipschitzLoss(k=0.5, reduction="sum"))
    huber_fn = loss.HuberFunc(reduction="sum")

    start_epoch = 1
    if continue_from is not None:
        if not os.path.exists(os.path.join(experiment_directory, ws.latent_codes_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.model_params_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.optimizer_params_subdir, continue_from + ".pth")):
            logging.warning('"{}" does not exist! Ignoring this argument...'.format(continue_from))
        else:
            logging.info('continuing from "{}"'.format(continue_from))

            model_epoch = ws.load_model_parameters(
                experiment_directory, continue_from, decoder
            )

            start_epoch = model_epoch + 1

            logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )

    use_pointwise_loss = get_spec_with_default(specs, "UsePointwiseLoss", False)
    pointwise_loss_weight = get_spec_with_default(specs, "PointwiseLossWeight", 0.0)

    use_pointpair_loss = get_spec_with_default(specs, "UsePointpairLoss", False)
    pointpair_loss_weight = get_spec_with_default(specs, "PointpairLossWeight", 0.0)

    logging.info("pointwise_loss_weight = {}, pointpair_loss_weight = {}".format(
        pointwise_loss_weight, pointpair_loss_weight))
    with open(os.path.join(experiment_directory, "loss.txt"), 'w') as f:
        f.write("Losses" + '\n')
    for epoch in range(start_epoch, num_epochs + 1):
        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        # batch_num = len(sdf_loader)
        epoch_loss = []
        epoch_sdf_loss = []
        epoch_pw_loss = []
        epoch_pp_loss = []
        epoch_bio_loss = []
        for bi, (sdf_data, indices) in enumerate(sdf_loader):
            batch_loss_sdf = 0.0
            batch_loss_pw = 0.0
            batch_loss_reg = 0.0
            batch_loss_pp = 0.0
            batch_loss_bio = 0.0
            optimizer_all.zero_grad()

            #### data["p_sdf"]: [b, frame_num, N, 4], data["t"]: [b, frame_num], indices: [b, ]
            # Process the input data
            p_sdf = sdf_data["p_sdf"].reshape(-1, 4)  # [b*frame_num*N, 4]
            t = sdf_data["t"].view(-1).unsqueeze(-1).repeat(1, N).view(-1).cuda()  # [b*frame_num*N, ]
            num_sdf_samples = p_sdf.shape[0]

            xyz = p_sdf[:, 0:3].view(-1, 3).cuda()  # [b*frame_num*N, 3]
            xyz.requires_grad = True
            sdf_gt = p_sdf[:, 3] # [b, 1, frame_num, N, 1]

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT).cuda()

            #### seq_index: [b, ]->[b, 1]->[b, frame_num*N]->[b*frame_num*N, ]
            seq_index = indices.unsqueeze(-1).repeat(1, frame_num * N).view(-1)
            #### frame_index: [frame_num, ]->[frame_num, 1]->[frame_num, N]
            # ->[frame_num*N, ]->[frame_num*N*b, ]
            frame_index = torch.IntTensor(list(range(frame_num))).unsqueeze(-1).repeat(1, N)\
                .view(-1).repeat(indices.shape[0])
            m_index = seq_index * frame_num + frame_index

            if initialize:
                cs_vecs = c_s[seq_index].cuda()  # [b*frame_num*N, Cs_size]
            else:
                cs_vecs = c_s(seq_index).cuda()  # [b*frame_num*N, Cs_size]
            cm_vecs = c_m(m_index).cuda()  # [b*frame_num*N, Cm_size]

            new_xyz, sdf_pred= decoder(xyz, t, cm_vecs, cs_vecs)
            if enforce_minmax:
                sdf_pred = torch.clamp(sdf_pred.squeeze(1), minT, maxT)

            #  sdf l1loss
            sdf_loss = loss_l1(sdf_pred, sdf_gt) / num_sdf_samples
            batch_loss_sdf += sdf_loss.item()
            batch_loss = sdf_loss

            # code regularization loss
            if do_code_regularization:
                index_nonED = torch.nonzero(t).squeeze()
                c_m_ = cm_vecs[index_nonED, :]
                cm_l2_size_loss = torch.sum(torch.norm(c_m_, dim=1))  # ED has no cm
                cm_reg_loss = cm_l2_size_loss / c_m_.shape[0]
                cs_l2_size_loss = torch.sum(torch.norm(cs_vecs, dim=1))
                cs_reg_loss = cs_l2_size_loss / num_sdf_samples
                reg_loss = cm_reg_loss + cs_reg_loss
                batch_loss_reg += reg_loss.item()
                batch_loss += code_reg_lambda * min(1.0, epoch / 100) * reg_loss.cuda()

            # pointwise loss
            if use_pointwise_loss:
                pw_loss = loss.apply_pointwise_reg(new_xyz, xyz, huber_fn, num_sdf_samples)
                batch_loss_pw += pw_loss.item()
                batch_loss += pw_loss.cuda() * pointwise_loss_weight * max(1.0, 10.0 * (1 - epoch / 100))
                # batch_loss += pw_loss.cuda() * pointwise_loss_weight

            # pointpair loss
            if use_pointpair_loss:
                lp_loss = loss.apply_pointpair_reg(new_xyz, xyz, loss_lp, batch_size, num_sdf_samples)
                batch_loss_pp += lp_loss.item()
                batch_loss += lp_loss.cuda() * pointpair_loss_weight * min(1.0, epoch / 100)
                # batch_loss += lp_loss.cuda() * pointpair_loss_weight

            batch_loss.backward()
            logging.debug("sdf_loss = {:.9f}, reg_loss = {:.9f}, pw_loss = {:.9f}, pp_loss = {:.9f}".format(
                batch_loss_sdf, batch_loss_reg, batch_loss_pw, batch_loss_pp))

            epoch_sdf_loss.append(batch_loss_sdf)
            epoch_pw_loss.append(batch_loss_pw)
            epoch_pp_loss.append(batch_loss_pp)
            epoch_bio_loss.append(batch_loss_bio)
            epoch_loss.append(batch_loss.item())

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

            # release memory
            del xyz, t, new_xyz, sdf_pred, sdf_loss, batch_loss_sdf, batch_loss_reg, batch_loss_pp, batch_loss_pw, batch_loss

        epoch_info = "epoch {}, total_loss = {:.6f}, sdf_loss = {:.6f}, pw_loss = {:.6f}, pp_loss = {:.6f}".format(
            epoch,
            sum(epoch_loss) / len(epoch_loss),
            sum(epoch_sdf_loss) / len(epoch_sdf_loss),
            sum(epoch_pw_loss) / len(epoch_pw_loss),
            sum(epoch_pp_loss) / len(epoch_pp_loss))
        print(epoch_info)
        with open(os.path.join(experiment_directory, "loss.txt"), 'a') as f:
            # f.write(epoch_info + '\n')
            f.write("{:.6f}".format(sum(epoch_loss) / len(epoch_loss)) + '\n')

        if epoch in checkpoints:
            save_checkpoints(epoch)
            if save_results:
                logging.info("Start reconstructing...")
                resolution = 64
                max_batch = int(2 ** 17)
                reconstruction_dir = os.path.join(
                    args.experiment_directory, "Reconstructions_training", "%04d"%epoch
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

                seqfiles = deep_sdf.dataset.get_instance_filenames(args.data_source, train_split)
                for ii, npz in enumerate(tqdm.tqdm(seqfiles)):
                    phase_list = os.listdir(npz)
                    for phase_idx in range(len(phase_list)):
                        full_filename = os.path.join(npz, phase_list[phase_idx])
                        file_name = os.path.split(npz)[1] + "_" + os.path.splitext(phase_list[phase_idx])[0]
                        # print("reconstruct: " + file_name)
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

                        if initialize:
                            c_s_vec = c_s[ii, :].unsqueeze(0)  # [1, Cs_size]
                        else:
                            c_s_vec = c_s(torch.LongTensor([ii]))
                        c_m_vecs = c_m(torch.LongTensor([ii * frame_num + phase_idx]))  # [1, Cm_size]
                        phase = torch.FloatTensor([phase_idx / (frame_num - 1)]).unsqueeze(0)  # [1, 1]
                        with torch.no_grad():
                            deep_sdf.mesh.create_mesh_4dsdf(
                                decoder, c_s_vec, c_m_vecs, phase, mesh_filename, motion_filename,
                                N=resolution, max_batch=max_batch, offset=offset, scale=scale, Ti=Ti)

                        if not os.path.exists(os.path.dirname(latent_filename)):
                            os.makedirs(os.path.dirname(latent_filename))

                        if initialize:
                            torch.save(c_s.detach().cpu(), latent_filename + "_cs.pth")
                        else:
                            torch.save(c_s.state_dict(), latent_filename + "_cs.pth")
                        torch.save(c_m.state_dict(), latent_filename + "_cm.pth")


if __name__ == "__main__":
    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train the model.")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.data_source, args.continue_from)
