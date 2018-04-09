#!/usr/bin/env python

import re

import h5py
with h5py.File('/n/home10/kcarnold/gajos_lab/Everyone/Data/COCO/trainval_feats.h5') as f:
    idx2imgid = f['image_ids'][:].tolist()

# open('data/coco/idx2id.txt','w').write('\n'.join(map(str, idx2imgid)))
# idx2imgid = [int(line.strip()) for line in open('data/coco/idx2id.txt')]

imgid2idx = {imgid: idx for idx, imgid in enumerate(idx2imgid)}

fname_regex = re.compile(r'COCO_(train|val)...._(\d+).jpg')

for subset in ['train', 'val']:
    with open(f'{subset}.txt.src', 'r') as in_fnames, open(f'{subset}.txt.src-idx', 'w') as out_indices:
        for line in in_fnames:
            match = fname_regex.match(line.strip())
            imgid = int(match.group(2).lstrip('0'))
            idx = imgid2idx[imgid]
            out_indices.write(f'{idx}\n')
