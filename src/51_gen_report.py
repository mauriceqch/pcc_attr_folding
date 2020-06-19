import argparse
import os
import json
from pyntcloud import PyntCloud
from utils import mpeg_parsing
from glob import glob
from importlib import import_module


def find(folder, pattern):
    files = glob(os.path.join(folder, pattern))
    assert len(files) == 1
    return files[0]


def get_n_points(f):
    return len(PyntCloud.from_file(f).points)


def run(folder_path, point_size=1):
    assert os.path.exists(folder_path), f'{folder_path} does not exist'

    experiment_report = os.path.join(folder_path, 'report.json')
    if os.path.exists(experiment_report):
        print(f'{experiment_report} exists')
    else:
        print(f'processing {folder_path}')
        pc_log = find(folder_path, '*.bin.log')
        pcerror_result = find(folder_path, '*.pc_error')
        decoded_pc = find(folder_path, '*.ply.bin.decoded.ply')

        log_data = mpeg_parsing.parse_bin_log(pc_log)
        pcerror_data = mpeg_parsing.parse_pcerror(pcerror_result)

        pos_total_size_in_bytes = log_data['pos_bitstream_size_in_bytes']
        input_point_count = get_n_points(log_data['uncompressed_data_path'])
        pos_bits_per_input_point = pos_total_size_in_bytes * 8 / input_point_count
        color_bits_per_input_point = log_data['color_bitstream_size_in_bytes'] * 8 / input_point_count
        data = {
            'pos_total_size_in_bytes': pos_total_size_in_bytes,
            'pos_bits_per_input_point': pos_bits_per_input_point,
            'color_bits_per_input_point': color_bits_per_input_point,
            'input_point_count': input_point_count
        }
        data = {**data, **log_data, **pcerror_data}
        with open(experiment_report, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='51_gen_report.py', description='Produces a report.json for an MPEG test folder.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder_path', help='MPEG test folder.')
    parser.add_argument('--point_size', default=1, type=int)
    args = parser.parse_args()

    run(args.folder_path, args.point_size)
