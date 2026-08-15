"""Dataset helpers for variable-length cardiac sequences."""

import logging
from pathlib import Path

import numpy as np
import torch


def get_instance_filenames(data_load, split="train"):
    frames = []
    sequence_index = 0
    for dataset_name, sequences in data_load.get(split, {}).items():
        for sequence_name, sequence in sequences.items():
            instance_list = sequence["instance_list"]
            for phase, filename in enumerate(instance_list):
                frames.append({
                    "npz": filename,
                    "frame_num": len(instance_list),
                    "seq_idx": sequence_index,
                    "dataset": dataset_name,
                    "name": sequence_name,
                    "phase": phase,
                })
            sequence_index += 1
    return frames


def get_sequences(data_load, split="test"):
    sequences_out = []
    sequence_index = 0
    for dataset_name, sequences in data_load.get(split, {}).items():
        for sequence_name, sequence in sequences.items():
            instances = sequence["instance_list"]
            sequences_out.append({
                "dataset": dataset_name,
                "name": sequence_name,
                "seq_idx": sequence_index,
                "frame_num": len(instances),
                "frame_list": [
                    {"npz": filename, "phase": phase}
                    for phase, filename in enumerate(instances)
                ],
            })
            sequence_index += 1
    return sequences_out


def _sample_rows(tensor, count):
    if tensor.shape[0] == 0:
        raise ValueError("Cannot sample an empty point set")
    indices = torch.randint(tensor.shape[0], (count,))
    return tensor.index_select(0, indices)


def get_sdf_samples(filename, subsample=None):
    with np.load(filename) as archive:
        if subsample is None:
            return {key: archive[key] for key in archive.files}
        pos = torch.from_numpy(archive["pos"]).float()
        neg = torch.from_numpy(archive["neg"]).float()
        pos = pos[torch.isfinite(pos).all(dim=1)]
        neg = neg[torch.isfinite(neg).all(dim=1)]
        positive_count = subsample // 2
        sample = torch.cat([
            _sample_rows(pos, positive_count),
            _sample_rows(neg, subsample - positive_count),
        ], dim=0)
        sample = sample[torch.randperm(sample.shape[0])]
        t = float(np.asarray(archive["t"]).reshape(-1)[0])
    return sample, t


def get_sdf_samples_test(filename, subsample=None):
    with np.load(filename) as archive:
        points = torch.from_numpy(archive["pcd"]).float()
        points = points[torch.isfinite(points).all(dim=1)]
        if subsample is not None:
            points = _sample_rows(points, subsample)
        t = float(np.asarray(archive["t"]).reshape(-1)[0])
    return points, t


class SDFSamples(torch.utils.data.Dataset):
    def __init__(self, data_load, subsample):
        self.subsample = subsample
        self.npzfiles = get_instance_filenames(data_load, "train")
        logging.info("Using %d frames", len(self.npzfiles))

    def __len__(self):
        return len(self.npzfiles)

    def __getitem__(self, idx):
        frame = self.npzfiles[idx]
        sample, t = get_sdf_samples(frame["npz"], self.subsample)
        return {
            "p_sdf": sample,
            "t": t,
            "frame_num": frame["frame_num"],
            "seq_name": frame["name"],
            "dataset_name": frame["dataset"],
            "phase": frame["phase"],
            "seq_idx": frame["seq_idx"],
            "inst_idx": idx,
        }, idx


def resolve_manifest_paths(data_load, repo_root):
    """Return a copy with repository-relative NPZ paths made absolute."""
    root = Path(repo_root)
    resolved = {"train": {}, "test": {}}
    for split, datasets in data_load.items():
        resolved[split] = {}
        for dataset_name, sequences in datasets.items():
            resolved[split][dataset_name] = {}
            for name, sequence in sequences.items():
                resolved[split][dataset_name][name] = {
                    "instance_list": [str(root / item) for item in sequence["instance_list"]]
                }
    return resolved
