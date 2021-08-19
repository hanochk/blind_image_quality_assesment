import json
import os
import pickle
from collections import namedtuple

"""
Process cutouts from annotator to blind quality pipeline (Eileen)
1. I've copied paste the outline from the annotator to a json and then merged them all to one pickle
2. used get_coutout_from_annotator_by_cutoutid_from_pickle() in Finder/sandbox to pull the cutouts 
"""
# path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline'
path = '/hdd/annotator_uploads/70e91f29c8f3f1147475c252c4e369c6/blindImages' #'/hdd/blind_raw_images/ee40624'
Point = namedtuple('Point', ['x', 'y'])

outline_dict = dict()
for root, _, filenames in os.walk(path):
    filenames.sort()
    for file in filenames:
        f, ext = os.path.splitext(file)
        if ext == '.json':
            with open(os.path.join(path, file)) as f:
                data_json = json.load(f)

            fname = file.split('/')[-1].split('.json')[0] + '.png'
            # fname = os.path.splitext(file)[0]
            # outline_dict[file.split('.')[0]] = [Point(val['x'], val['y']) for val in data_json['coords']]
            outline_dict.update({fname: {'contour': [Point(val['x'], val['y']) for val in data_json['contour']]}})

with open(os.path.join(path, 'merged_outlines_json.pkl'), 'wb') as f:
    pickle.dump(outline_dict, f)


# cutout_2_outline_n_meta['52f2e22b-a21a-5575-8d8b-a6a3a761f95d']['outline']
# [Point(x=2822.477040710449, y=830.7151653921005), Point(x=2767.6746734619137, y=844.7912797447589)