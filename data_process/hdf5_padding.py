# pad hdf5 data
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import convert_all as crd
import argparse


parser = argparse.ArgumentParser(description="Pad hdf5 data")
parser.add_argument(
    "-ins", "--input_directories", type=str, nargs="+", help="directories to pad"
)
parser.add_argument(
    "-out",
    "--output_directory",
    type=str,
    help="output directory to save the padded data",
)

args = parser.parse_args()

in_dirs = args.input_directories
out_dir = args.output_directory

os.makedirs(out_dir, exist_ok=True)

# get the max length of the data
max_length = 0
for d in in_dirs:
    max_length = max(max_length, len(crd.hdf5_to_dict(d + "/episode_0.hdf5")["action"]))
print("max length: ", max_length)

# pad the data
cnt = 0
for d in in_dirs:
    hdf5_files = crd.get_files_name_by_suffix(d, ".hdf5")
    print(f"Found {len(hdf5_files)} hdf5 files in {d}")
    # print(hdf5_files[0])
    for file in hdf5_files:
        data = crd.flatten_dict(crd.hdf5_to_dict(f"{d}/{file}"), prefix="/")
        # print(data.keys())
        crd.save_dict_to_hdf5(
            data,
            f"{out_dir}/episode_{cnt}.hdf5",
            max_length,
        )
        cnt += 1
print(f"Saved {cnt} hdf5 files to {out_dir}")
