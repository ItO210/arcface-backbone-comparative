import os
from PIL import Image
import numpy as np
import mxnet as mx
from tqdm import tqdm

# Run this to turn record files to folders.

# Paths to the MXNet record files
rec_file = "path/to/train.rec"
idx_file = "path/to/train.idx"
output_dir = 'VGG2'

os.makedirs(output_dir, exist_ok=True)

record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')

record_indices = list(record.keys)
total_records = len(record_indices)

for i in tqdm(record_indices, total=total_records, desc="Extracting images"):
    try:
        s = record.read_idx(i)
        if s is None:
            print(f"Warning: Empty record at index {i}")
            continue

        header, img_buf = mx.recordio.unpack(s)
        if not img_buf:
            print(f"Warning: Empty image buffer at index {i}")
            continue

        label_raw = header.label
        if isinstance(label_raw, (list, tuple, np.ndarray)):
            label = int(label_raw[0])
        else:
            label = int(label_raw)

        img_array = mx.image.imdecode(img_buf).asnumpy()
        img = Image.fromarray(img_array)

        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        save_path = os.path.join(label_dir, f'{i}.jpg')
        img.save(save_path)

    except Exception as e:
        print(f"Error processing index {i}: {e}")
