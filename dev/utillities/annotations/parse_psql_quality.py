import os
import pandas as pd
import tqdm
import numpy as np
""""
psql -P pager=off -U read_only_user -d blind_dev -c " select grades.media_id, users.username, grades.image_quality_grade, grades.created_date
from grades join users ON (users.id = grades.user_id) where created_date is not null and created_date between '2021-05-08 00:00:00' and '2021-05-16 00:00:00' order by media_id;" > annotators_quality.txt(fis
"""
path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\DataBaseInfo\blind_quality'
txt_file = 'annotators_quality.txt'


if 0:
    year_start = 2020 #1900
    month_start = 7#1
else:
    year_start = 2021
    month_start = 5


csv_file = txt_file.split('.txt')[0] + str(month_start) + '_' + str(year_start) + '.csv'
extract_feilds_from_file = True

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
                                # if ind >0:
                                dictionary.update({field: fields[ind]})
                                # else:
                                #     dictionary.update({"cutout_id": fields[ind]}) # h"cutout_id" has to be the 1st since many call it id

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
            f = [f.strip() for f in fields]
            isalpha_ =  [str.isalpha(f1.replace("_","")) for f1 in f] #
            if np.array(isalpha_).astype('int').prod() == 1: # catch the header only
                filds_in_file = [i.strip() for i in fields]

all_cutout_id_list = np.array(cutout_id_list)
# np.savetxt(os.path.join(path, cutout_id_only_txt_file), all_cutout_id_list, fmt='%s')
df_cutout_map2 = pd.DataFrame(lst)
df_cutout_map2.to_csv(os.path.join(path, csv_file), index=False)
csv_file_per_ex = 'csv_file_per_ex' + csv_file
users_acm = dict.fromkeys(df_cutout_map2.username.unique())
media_list = list()
for mu in df_cutout_map2.media_id.unique():
    dictionary = dict()
    for user in df_cutout_map2[df_cutout_map2.media_id == mu].username:
        annot = df_cutout_map2[df_cutout_map2.media_id == mu][df_cutout_map2.username == user].image_quality_grade.item()
        dictionary.update({user: annot})
    media_list.append(dictionary)

df_ready_to_hist = pd.DataFrame(media_list)
df_ready_to_hist.to_csv(os.path.join(path, csv_file_per_ex), index=False)