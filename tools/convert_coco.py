#!/usr/bin/env python

"""
Convert MS COCO annotations JSON files into line-oriented src and tgt files for OpenNMT.
"""

import json
from nltk.tokenize import ToktokTokenizer
tokenizer = ToktokTokenizer()

for subset in ['train', 'val']:
    with open('captions_{}2014.json'.format(subset)) as f:
        data = json.load(f)
    
    id2img = {img['id']: img['file_name'] for img in data['images']}
    
    with open(f'{subset}.txt.src', 'w') as out_src, open(f'{subset}.txt.tgt', 'w') as out_tgt:
        for annotation in sorted(data['annotations'], key=lambda x: x['image_id']):
            out_src.write('{}\n'.format(id2img[annotation['image_id']] ))
            cap = annotation['caption'].strip().replace('\n', ' ')
            cap_tok = tokenizer.tokenize(cap, return_str=True).lower()
            assert '\n' not in cap_tok
            out_tgt.write('{}\n'.format(cap_tok))
