import argparse
import base64
import numpy as np
import csv
import sys
import h5py
import tqdm

"""
Converts pretrained RCNN vectors from TSV to HDF5

Vectors come from:
https://github.com/peteanderson80/bottom-up-attention

Based on their tool:
https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/read_tsv.py
"""


csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

feature_length = 2048
num_objs = 36

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('tsv_file',
                        help="TSV input filename")
    parser.add_argument('h5_file',
                        help="HDF5 output filename")
    parser.add_argument('--num_imgs', type=int, default=123287,
                        help="Number of images in the TSV file")
    opt = parser.parse_args()


    num_imgs = opt.num_imgs
    h5file = h5py.File(opt.h5_file, 'w')
    img_features = h5file.create_dataset(
        'image_features', (num_imgs, num_objs, feature_length), 'f')
    img_ids = h5file.create_dataset(
        'image_ids', (num_imgs,), 'i')

    with open(opt.tsv_file, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for idx, item in enumerate(tqdm.tqdm(reader, total=num_imgs, mininterval=1)):
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodebytes(item[field].encode('latin1')), 
                      dtype=np.float32).reshape((item['num_boxes'],-1))

            img_ids[idx] = item['image_id']
            img_features[idx] = item['features']

    h5file.close()
