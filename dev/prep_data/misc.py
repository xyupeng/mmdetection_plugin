import os

def main():
    root = './data/waymo'
    for split in ['training', 'validation']:
        split_dir = os.path.join(root, split)
        filename = 'val.txt' if split == 'validation' else 'train.txt'
        src_path = os.path.join(split_dir, filename)
        with open(src_path, 'r') as f:
            lines = f.readlines()
        lines = sorted(lines)
        out_path = os.path.join(split_dir, 'sorted.txt')
        assert not os.path.isfile(out_path)
        with open(out_path, 'w') as f:
            f.writelines(lines)

main()