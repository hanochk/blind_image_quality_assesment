import os
import pandas as pd
import tqdm
import numpy as np
# pipeline of processing : sql query into textfile with the related fields
# 1. run parse_psql_res_2_csv.py in blind_quality_svm project it will parse and create .csv with the {"cutout_id": fields[0], "media_id": fields[1], "create_date": fields[2],  "path": fields[3]}
# 2. take the csv outcome and run get_from_annotator.py in Finders/sandbox project it will extract from annotator the unit no related to the cutout_id
# 3. merge the csv outcome merge_fleet_unit_info_annotator_images.py to get the RIGHT location of the unit came from the marketing xls fleet info
path = '/hdd/hanoch/data/database'
# txt_file = 'omri_outfulljoin.txt'
# txt_file = 'cutouts_path_all_db.txt'
txt_file = 'query_with_quality.txt'
extract_feilds_from_file = True
if 0:
    year_start = 2020 #1900
    month_start = 7#1
else:
    year_start = 1900
    month_start = 1

cutout_id_only_txt_file = txt_file.split('.txt')[0] + str(month_start) + '_' + str(year_start) + '_txt.txt'
csv_file = txt_file.split('.txt')[0] + str(month_start) + '_' + str(year_start) + '.csv'

lst = list()
cutout_id_list = list()
with open(os.path.join(path, txt_file), 'r') as cutouts_path_media_id:
    for line in tqdm.tqdm(cutouts_path_media_id):
        fields = line.split("|")
        # finds digit
        if any(char.isdigit() for char in line.split("|")[0].strip()):
            fields = [f.strip() for f in fields]
            if len(fields) >1:
                #filter by date
                date = fields[2]
                if 1: #(int(date.split('-')[0]))>(year_start-1) and int(date.split('-')[1])>(month_start-1):
                    try:
                        if extract_feilds_from_file:
                            dictionary = dict()
                            for ind, field in enumerate(filds_in_file):
                                if ind >0:
                                    dictionary.update({field: fields[ind]})
                                else:
                                    dictionary.update({"cutout_id": fields[ind]}) # h"cutout_id" has to be the 1st since many call it id

                            cutout_id_list.append(fields[0])
                            lst.append(dictionary)
                        else:
                            if len(fields) == 4:
                                dictionary = {"cutout_id": fields[0], "media_id": fields[1], "create_date": fields[2],  "path": fields[3]}
                                cutout_id_list.append(fields[0])
                            elif len(fields) == 3:
                                dictionary = {"cutout_id": fields[0], "media_id": fields[1], "path": fields[2]}
                                cutout_id_list.append(fields[0])
                            elif len(fields) == 2:
                                dictionary = {"media_id": fields[0], "path": fields[1]}
                            else:
                                raise ValueError("unknown option")

                            lst.append(dictionary)
                    except:
                        print(len(fields))
        else:
            if str.isalpha(fields[0].strip()): # catch the header only
                filds_in_file = [i.strip() for i in fields]

all_cutout_id_list = np.array(cutout_id_list)
np.savetxt(os.path.join(path, cutout_id_only_txt_file), all_cutout_id_list, fmt='%s')
df_cutout_map2 = pd.DataFrame(lst)
df_cutout_map2.to_csv(os.path.join(path, csv_file), index=False)
