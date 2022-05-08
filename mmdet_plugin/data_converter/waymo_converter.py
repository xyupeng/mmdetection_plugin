import os
from glob import glob
import time
import pickle

from ..utils.util import format_time
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2


class WaymoConverter:

    CAM_ID_TO_TYPE = {
        0: 'CAM_FRONT',
        1: 'CAM_FRONT_LEFT',
        2: 'CAM_FRONT_RIGHT',
        3: 'CAM_SIDE_LEFT',
        4: 'CAM_SIDE_RIGHT',
    }

    SPLIT_TO_ID = {
        'training': 0,
        'validation': 1,
        'testing': 2,
    }

    def __init__(self, root='./data/waymo/waymo_format'):
        self.root = root
        self.paths_dict = {
            'training': sorted(glob(os.path.join(root, 'training', '*.tfrecord'))),
            'validation': sorted(glob(os.path.join(root, 'validation', '*.tfrecord'))),
            'testing': sorted(glob(os.path.join(root, 'testing', '*.tfrecord'))),
        }

    def export_bbox_2d(self, split='training', num_segs=None, out_path=None):
        """
            create infos (list[dict]):
                infos[0]: {
                    'CAM_FRONT': {
                        'sample_idx' (str): 0000000  # 1000000 for validation
                        'cam_type' (str): 'CAM_FRONT'
                        'image_path' (str): image_0/0000000.png
                        'bboxes': ndarray(shape=[N, 4], dtype=np.float32); format: [x0, y0, x1, y1]
                        'labels': ndarray(shape=[N], dtype=np.int64)
                    },
                    'CAM_FRONT_LEFT':
                    'CAM_FRONT_RIGHT':
                    'CAM_SIDE_LEFT':
                    'CAM_SIDE_RIGHT':
                }

            class_type: {
                0: 'Vehicle',
                1: 'Pedestrian',
                2: 'Sign',
                3: 'Cyclist',
            }
        """
        assert out_path.endswith('.pkl')
        assert not os.path.isfile(out_path)
        assert os.path.isdir(os.path.dirname(out_path))

        print('==> Start exporting 2d bboxes...')
        t1 = time.time()
        split_idx = WaymoConverter.SPLIT_TO_ID[split]
        infos = []  # len == num_frames

        tfrecord_paths = self.paths_dict[split]
        if num_segs is not None:
            tfrecord_paths = tfrecord_paths[:num_segs]
        for seg_idx, path in enumerate(tfrecord_paths):
            print(f'==> Start seg {seg_idx}...')
            dataset = tf.data.TFRecordDataset(path, compression_type='')
            for frame_idx, data in enumerate(dataset):
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))

                info = {}
                for cam_labels in frame.camera_labels:

                    cam_id = cam_labels.name - 1  # enum/int: [0, 1, 2, 3, 4]
                    cam_type = WaymoConverter.CAM_ID_TO_TYPE[cam_id]
                    sample_idx = f'{split_idx:01d}{seg_idx:03d}{frame_idx:03d}'
                    image_path = f'image_{cam_id}/{sample_idx}.png'

                    bboxes, labels = [], []
                    for cam_label in cam_labels.labels:
                        bbox = [
                            cam_label.box.center_x - cam_label.box.length / 2,
                            cam_label.box.center_y - cam_label.box.width / 2,
                            cam_label.box.center_x + cam_label.box.length / 2,
                            cam_label.box.center_y + cam_label.box.width / 2
                        ]  # [x0, y0, x1, y1]
                        type = cam_label.type - 1  # enum/int: [0, 1, 2, 3]
                        bboxes.append(bbox)
                        labels.append(type)

                    bboxes = np.array(bboxes, dtype=np.float32)
                    labels = np.array(labels, dtype=np.int64)
                    info[cam_type] = {
                        'sample_idx': sample_idx,
                        'cam_type': cam_type,
                        'image_path': image_path,
                        'bboxes': bboxes,
                        'labels': labels,
                    }
                infos.append(info)

        with open(out_path, 'wb') as f:
            pickle.dump(infos, f)

        t2 = time.time()
        tot_time = format_time(t2 - t1)
        print(f'==> Done (time={tot_time}).')
