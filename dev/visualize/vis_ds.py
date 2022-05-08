import argparse
import os

import numpy as np
from mmcv import Config, DictAction
from mmdet.apis import init_random_seed
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.apis import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--idx', type=int, default=0, help='sample_id')
    parser.add_argument('--split', required=True, help='[train, val, test, vis]')
    parser.add_argument('--loader', action='store_true', help='build dataloader')
    parser.add_argument('--vis-3d', action='store_true', help='visualize points')
    parser.add_argument('--vis-2d', action='store_true', help='visualize images')
    parser.add_argument('--debug', action='store_true', help='debug dataset')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)  # already handled custom_imports
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename, 'vis')
    os.makedirs(cfg.work_dir, exist_ok=True)

    # set random seeds
    seed = init_random_seed(args.seed)
    set_random_seed(seed)
    cfg.seed = seed

    return cfg


def vis_2d(input_dict, sample_idx, cfg):
    from PIL import Image
    from mmdet.core.visualization.image import imshow_det_bboxes

    if input_dict['img_prefix'] is not None:
        filepath = os.path.join(input_dict['img_prefix'], input_dict['img_info']['filename'])
    else:
        filepath = input_dict['img_info']['filename']
    # img = np.array(Image.open(filepath))  # array(shape=[H, W, 3], dtype=np.uint8)

    bboxes = input_dict['ann_info']['bboxes']
    labels = input_dict['ann_info']['labels']
    save_path = os.path.join(cfg.work_dir, f'sample_{sample_idx}.png')
    img = imshow_det_bboxes(filepath, bboxes, labels, show=True, out_file=save_path)


def main():
    args = parse_args()
    cfg = get_cfg(args)

    ds = build_dataset(cfg.data.get(args.split))
    if not hasattr(ds, 'data_infos'):
        ds = ds.dataset
    sample_idx = args.idx
    img_info = ds.data_infos[sample_idx]  # before get_data_info()
    ann_info = ds.get_ann_info(sample_idx)
    input_dict = dict(img_info=img_info, ann_info=ann_info)
    input_dict = ds.pre_pipeline(input_dict)
    x = ds[sample_idx]  # after pipeline
    if args.debug:
        import pdb; pdb.set_trace()

    if args.vis_2d:
        vis_2d(input_dict, sample_idx, cfg)

    if args.loader:
        data_loader = build_dataloader(
            ds,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )
        y = next(iter(data_loader))
        import pdb; pdb.set_trace()


# python dev/visualize/vis_ds.py configs/_base_/datasets/coco_detection.py --split test \
# [--vis-2d --loader --work-dir ./work_dir/debug]
if __name__ == '__main__':
    main()
