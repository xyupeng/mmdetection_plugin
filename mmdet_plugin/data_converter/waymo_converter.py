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

    def export_bbox_2d(self, root='./data/waymo', split='training', num_segs=None, out_path=None):
        """
            create infos from kitti_format 
            infos (list[dict]):
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

        tfrecord_paths = sorted(glob(os.path.join(root, 'waymo_format', split, '*.tfrecord'))),
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

    def export_bbox_2d_ply(self, root='./data/waymo', split='training'):
        """
            create infos from xiaofei's ply_format
            infos: (list[dict]):
                infos[0]: {
                        'cam_type' (str): 'CAM_FRONT'
                        'image_path' (str): validation_0000/{seg_name}_{frame_idx}_{cam_id}.png  # cam_id within {1, 2, 3, 4, 5}
                        'bboxes': ndarray(shape=[N, 4], dtype=np.float32); format: [x0, y0, x1, y1]
                        'labels': ndarray(shape=[N], dtype=np.int64)
                }
                labels: {
                    0: 'Vehicle',
                    1: 'Pedestrian',
                    2: 'Sign',
                    3: 'Cyclist',
                }
        """
        print('==> Start exporting 2d bboxes...')
        t1 = time.time()

        sample_path = os.path.join(root, 'ply_format', split, 'sorted.txt')
        with open(sample_path, 'r') as f:
            frame_list = [line.strip() for line in f.readlines()]
            import pdb; pdb.set_trace()
        prev_idx, cur_idx = 0, 0

        split_dir = os.path.join(root, 'waymo_format', split)
        sub_dirs = [os.path.join(split_dir, fname) for fname in os.listdir(split_dir) if os.path.isdir(split_dir, fname)]
        import pdb; pdb.set_trace()

        for sub_dir in sub_dirs:
            infos = []
            sub_dirname = os.path.basename(sub_dir)
            out_path = os.path.join(os.path.dirname(sub_dir), f'{sub_dirname}.pkl')
            assert not os.path.isfile(out_path)

            tfrecord_paths = sorted(glob(os.path.join(sub_dir, '*.tfrecord')))
            for seg_idx, path in enumerate(tfrecord_paths):
                print(f'==> Start {sub_dirname}/seg_{seg_idx}...')
                seg_name = os.path.basename(path).splitext()[0]

                # get 3D segmentation task frames for this segment
                while seg_name in frame_list[cur_idx]:
                    cur_idx += 1
                    if cur_idx == len(frame_list):
                        break
                frame_ids = frame_list[prev_idx: cur_idx]
                prev_idx = cur_idx
                frame_ids = [int(line.split('_')[-1].split('.')[0]) for line in frame_list]
                frame_ids = sorted(frame_ids)
                import pdb; pdb.set_trace()

                # generate info dict for this segment
                dataset = tf.data.TFRecordDataset(path, compression_type='')
                for frame_idx, data in enumerate(dataset):
                    if frame_idx not in frame_ids:
                        continue

                    frame = dataset_pb2.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    for cam_labels in frame.camera_labels:

                        cam_id = cam_labels.name - 1  # enum/int: [0, 1, 2, 3, 4]
                        cam_type = WaymoConverter.CAM_ID_TO_TYPE[cam_id]
                        image_path = f'{sub_dirname}/{seg_name}_{frame_idx}_{cam_id + 1}.png'

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

                        info = {
                            'cam_type': cam_type,
                            'image_path': image_path,
                            'bboxes': bboxes,
                            'labels': labels,
                        }
                        import pdb; pdb.set_trace()
                        infos.append(info)

            with open(out_path, 'wb') as f:
                pickle.dump(infos, f)

        t2 = time.time()
        tot_time = format_time(t2 - t1)
        print(f'==> Done (Total time={tot_time}).')
