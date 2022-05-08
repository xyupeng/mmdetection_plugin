import os
from collections import defaultdict

import mmcv
import numpy as np
from PIL import Image
import pickle

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class WaymoDet2D(CustomDataset):
    def __init__(self, load_interval=1, **kwargs):
        self.load_interval = load_interval
        super(WaymoDet2D, self).__init__(**kwargs)

    def load_annotations(self, ann_path):
        """
        Args:
            ann_path: absolute path to info file

        Returns:
            data_infos (list[dict]):  # first order: cam_type; second: sample_idx
                data_infos[0]: {
                    'sample_idx' (str): 0000000  # 1000000 for validation
                    'cam_type' (str): cam_type  # ('CAM_FRONT', 'CAM_FRONT_LEFT'...)
                    'image_path' (str): image_0/0000000.png
                    'filename' (str): the same as 'image_path'
                    'bboxes': ndarray(shape=[N, 4], dtype=np.float32); format: [x0, y0, x1, y1]
                    'labels': ndarray(shape=[N], dtype=np.int64)
                },

        """
        data_infos = []
        with open(ann_path, 'rb') as f:
            infos = pickle.load(f)

        # first order: cam; second: sample_idx
        for cam_type in infos[0].keys():
            for info_dict in infos:
                new_dict = info_dict[cam_type].copy()
                new_dict['filename'] = new_dict['image_path']
                data_infos.append(new_dict)

        return data_infos

    def get_ann_info(self, idx):
        """
        Returns:
            ann: {
                'bboxes': ndarray(shape=[N, 4], dtype=np.float32); format: [x0, y0, x1, y1]
                'labels': ndarray(shape=[N], dtype=np.int64)
            }
        """
        info_dict = self.data_infos[idx]
        ann = dict(
            bboxes=info_dict['bboxes'],
            labels=info_dict['labels'],
        )
        return ann

    def _filter_imgs(self):
        """Filter images by interval."""
        interval_inds = list(range(0, len(self.data_infos), self.load_interval))
        valid_inds = []
        if self.filter_empty_gt:
            for idx in interval_inds:
                info_dict = self.data_infos[idx]
                if len(info_dict['labels']) > 0:
                    valid_inds.append(idx)
        return valid_inds

    def _set_group_flag(self):
        pass

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
