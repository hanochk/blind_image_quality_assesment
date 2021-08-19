import pandas as pd
import subprocess
import os
import tqdm
path_src = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/fitjar/res/tiles'
pash_dest = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/fitjar/res/tiles/mod_suffix_per_class'
xls_annot = '/hdd/hanoch/data/cages_holdout/annotated_images_eileen_fitjar/res/csv/list_ready_cage_fitjar_blind_test_results.xlsx'
df_annot = pd.read_excel(xls_annot, index_col=False, engine='openpyxl',)
pattern_field = '.png_cls_'
i = 0
for idx, cu in enumerate(tqdm.tqdm(df_annot.cutout_id)):
    label_dest = df_annot[df_annot.cutout_id == cu].Eileen.item().lower()
    # print(idx, cu)
    file_full_path = subprocess.getoutput('find ' + path_src + ' -iname ' + '"*' + cu + '*"')
    # print(file_full_path)
    if file_full_path == '':
        print('File not found in repo')
    else:
        if '\n' in file_full_path:
            tiles_list = file_full_path.split('\n')
        for tile_path in tiles_list:
            tiler_dest = os.path.split(tile_path)[-1].replace("_bad_", "_" + label_dest + "_")
            process = subprocess.Popen(['cp', '-p', tile_path, pash_dest + '/' + tiler_dest],
                                       stderr=subprocess.STDOUT,
                                       stdout=subprocess.PIPE)
            if process.poll() != 0:
                i = i+1 # dummy
                # print('hueston')
