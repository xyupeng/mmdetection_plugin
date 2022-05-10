import os

import mmcv
import numpy as np
from PIL import Image
import pickle

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class WaymoDet2D(CustomDataset):
    def __init__(self, info_format, num_cams=5, load_interval=1, **kwargs):
        assert info_format in ['ply', 'kitti']

        self.info_format = info_format
        self.num_cams = num_cams
        self.load_interval = load_interval
        super(WaymoDet2D, self).__init__(**kwargs)

    def load_annotations_ply(self, ann_path):
        """
        Args:
            ann_path: absolute path to info file

        Returns:
            data_infos (list[dict]):  #  lexicographic order by {seg_name}_{frame_idx}_{cam_id}
                data_infos[0]: {
                    'cam_type' (str): cam_type  # ('CAM_FRONT', 'CAM_FRONT_LEFT'...)
                    'image_path' (str): validation_0000/{seg_name}_{frame_idx}_{cam_id}.png  # cam_id within {1, 2, 3, 4, 5}
                    'filename' (str): the same as 'image_path'
                    'bboxes': ndarray(shape=[N, 4], dtype=np.float32); format: [x0, y0, x1, y1]
                    'labels': ndarray(shape=[N], dtype=np.int64)
                },

        """
        data_infos = []
        with open(ann_path, 'rb') as f:
            infos = pickle.load(f)

        for info_dict in infos:
            info_dict['filename'] = info_dict['image_path']
            data_infos.append(info_dict)

        return data_infos

    def load_annotations_kitti(self, ann_path):
        """
        Args:
            ann_path: absolute path to info file

        Returns:
            data_infos (list[dict]):  # first order: sample_idx; second: cam_type
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
        for info_dict in infos:
            for cam_type in info_dict.keys():
                new_dict = info_dict[cam_type]
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

    def load_annotations(self, ann_path):
        if self.info_format == 'ply':
            return self.load_annotations_ply(ann_path)
        elif self.info_format == 'kitti':
            return self.load_annotations_kitti(ann_path)
        else:
            raise ValueError

    def _filter_imgs(self):
        """Filter images by interval."""
        inds = []
        for k in range(0, len(self.data_infos) // self.num_cams, self.load_interval):
            inds.extend(list(range(k * self.num_cams, (k+1) * self.num_cams)))

        valid_inds = []
        if self.filter_empty_gt:
            for idx in inds:
                info_dict = self.data_infos[idx]
                if len(info_dict['labels']) > 0:
                    valid_inds.append(idx)
        else:
            valid_inds = inds
        return valid_inds

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
