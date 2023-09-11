import argparse
import json
import os
from mesh_to_sdf.sdf_process import transform_to_canonical


def process(data_dir, source_dir, source_name, class_name, test_sampling):
    instance_list = []
    for instance_name in split[args.source_name][args.class_name]:
        instance_list += [instance_name]
    target_dir = os.path.join(data_dir,
                              source_name,
                              class_name)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)  # data/LVV/train

    for instance in instance_list:
        base_path = os.path.join(source_dir, instance)
        transform_to_canonical(base_path, instance,
                           target_dir, test_sampling=test_sampling)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to "
                    + "a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source_dir",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--source_name",
        "-sn",
        dest="source_name",
        required=True,
        help="The name to use for the data source. If unspecified, it defaults to the "
             + "directory name.",
    )
    arg_parser.add_argument(
        "--class_name",
        "-cn",
        dest="class_name",
        required=True,
        help="The name to use for the class source. If unspecified, it defaults to the "
             + "directory name.",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )

    args = arg_parser.parse_args()
    args.test_sampling = False
    with open(args.split_filename, "r") as f:
        split = json.load(f)

    process(args.data_dir, args.source_dir, args.source_name, args.class_name, args.test_sampling)