from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

download_path = Path('/home/minghach/Data/CMH/LPN/dataset/cvusa/CVUSA_ori/')
train_split = download_path / 'splits/train-19zl.csv'
train_save_path = download_path / 'train'

if not train_save_path.is_dir():
    train_save_path.mkdir()
    (train_save_path / 'street').mkdir()
    (train_save_path / 'satellite').mkdir()

with open(train_split) as fp:
    lines = fp.readlines()

for line in tqdm(lines, desc="Processing training set"):
    filename = line.split(',')
    src_path = download_path / filename[0]

    # stem = Path(filename[0]).stem
    # print(stem)
    # assert(0)
    dst_path = train_save_path / 'satellite' / Path(filename[0]).stem
    if not dst_path.is_dir():
        dst_path.mkdir()
    copyfile(src_path, dst_path / Path(filename[0]).name)

    src_path = download_path / filename[1].strip()
    dst_path = train_save_path / 'street' / Path(filename[1]).stem
    if not dst_path.is_dir():
        dst_path.mkdir()
    copyfile(src_path, dst_path / Path(filename[1]).name)

val_split = download_path / 'splits/val-19zl.csv'
val_save_path = download_path / 'val'

if not val_save_path.is_dir():
    val_save_path.mkdir()
    (val_save_path / 'street').mkdir()
    (val_save_path / 'satellite').mkdir()

with open(val_split) as fp:
    lines = fp.readlines()

for line in tqdm(lines, desc="Processing validation set"):
    filename = line.split(',')
    src_path = download_path / filename[0]
    dst_path = val_save_path / 'satellite' / Path(filename[0]).stem
    if not dst_path.is_dir():
        dst_path.mkdir()
    copyfile(src_path, dst_path / Path(filename[0]).name)

    src_path = download_path / filename[1].strip()
    dst_path = val_save_path / 'street' / Path(filename[1]).stem
    if not dst_path.is_dir():
        dst_path.mkdir()
    copyfile(src_path, dst_path / Path(filename[1]).name)
