import os
import sys
sys.path.append('.')
import argparse
from mmdet_plugin.data_converter.waymo_converter import WaymoConverter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', required=True, choices=['ply', 'kitti'])
    parser.add_argument('--split', required=True, choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_segs', type=int, default=None, help='number of segments')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    split = args.split
    num_segs = args.num_segs
    root = './data/waymo'

    converter = WaymoConverter(root=os.path.join(root, 'waymo_format'))
    if args.format == 'kitti':
        out_path = os.path.join(root, f'kitti_format/waymo_det2d_infos_{split}_seg_{num_segs}.pkl')
        converter.export_bbox_2d(root=root, split=split, num_segs=num_segs, out_path=out_path)
    elif args.format == 'ply':
        converter.export_bbox_2d_ply(root='./data/waymo', split=split)
    else:
        raise ValueError


main()
