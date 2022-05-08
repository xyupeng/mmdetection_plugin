## Prepare data
```
# waymo 2d bbox
python dev/prep_data/waymo_bbox_2d.py --split train [--num_segs 20]
```


## Visualize
```
# visualize COCO validation split
python dev/visualize/vis_ds.py configs/_base_/datasets/coco_detection.py --split val --idx 0 [--vis-2d --debug] [--loader --work-dir ./work_dir/debug]
```