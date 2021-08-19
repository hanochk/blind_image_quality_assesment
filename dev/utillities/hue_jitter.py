from PIL import Image
from io import BytesIO
from collections import namedtuple
import tqdm
import numpy as np
import os
import pandas as pd
import subprocess
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm

# domain_eileen_path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\blind_quality_svm\EileenQuality\comparing domains_good_quality'
# domain_eileen_path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\blind_quality_svm\no_gamma_holdout\eileen_th_0p18_softmap\tiles'
domain_eileen_path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/cutout_data/EileenQuality_needtoRunblindFinder_4_outline/tiles'
# domain_b_path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\blind_quality_svm\EileenQuality\comparing domains_good_quality\blind_q_holdout'
# domain_b_path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Data\blind_quality_svm\EileenQuality\comparing domains_good_quality\tiles\blind_q_holdout'
domain_b_path = '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data/train/good'

results_path = '/hdd/hanoch/runmodels/img_quality'
filenames_d_eileen = [os.path.join(domain_eileen_path, x) for x in os.listdir(domain_eileen_path)
             if x.endswith('png')]

filenames_b = [os.path.join(domain_b_path, x) for x in os.listdir(domain_b_path)
             if x.endswith('png')]

r_e = np.array([])
g_e = np.array([])
b_e = np.array([])
for file in tqdm.tqdm(filenames_d_eileen):
    img = Image.open(file).convert("RGB")
    image_rgb = np.asarray(img)
    image = mcolors.rgb_to_hsv(image_rgb)

    r_e = np.append(r_e, image[:, :, 0].ravel())[:, np.newaxis]
    g_e = np.append(g_e, image[:, :, 1].ravel())[:, np.newaxis]
    b_e = np.append(b_e, image[:, :, 2].ravel())[:, np.newaxis]

r_b = np.array([])
g_b = np.array([])
b_b = np.array([])
for file in tqdm.tqdm(filenames_b):
    img = Image.open(file).convert("RGB")
    image_rgb = np.asarray(img)
    image = mcolors.rgb_to_hsv(image_rgb)

    r_b = np.append(r_b, image[:, :, 0].ravel())[:, np.newaxis]
    g_b = np.append(g_b, image[:, :, 1].ravel())[:, np.newaxis]
    b_b = np.append(b_b, image[:, :, 2].ravel())[:, np.newaxis]


fig, ax = plt.subplots(3, 1)
ax[0].title.set_text('H histogram comparison of good quality images: b:Eileen;g:holdout')
# fig.clf()

a_e = np.histogram(r_e, bins=100)
a_b = np.histogram(r_b, bins=100)
ax[0].set_xlabel('pixel level')
ax[0].set_ylabel('Frequency')
bins_loc_e = (a_e[1][0:-1] + a_e[1][1:]) / 2
bins_loc_b = (a_b[1][0:-1] + a_b[1][1:]) / 2
ax[0].semilogy(bins_loc_e, a_e[0] / sum(a_e[0]), 'b-', bins_loc_b, a_b[0] / sum(a_b[0]), 'g-')
ax[0].grid()



a_e = np.histogram(g_e, bins=100)
a_b = np.histogram(g_b, bins=100)
ax[1].set_xlabel('pixel level')
ax[1].set_ylabel('Frequency')
bins_loc_e = (a_e[1][0:-1] + a_e[1][1:]) / 2
bins_loc_b = (a_b[1][0:-1] + a_b[1][1:]) / 2
ax[1].semilogy(bins_loc_e, a_e[0] / sum(a_e[0]), 'b-', bins_loc_b, a_b[0] / sum(a_b[0]), 'g-')
ax[1].title.set_text('S histogram comparison of good quality images: b:Eileen;g:holdout')

ax[1].grid()

a_e = np.histogram(b_e, bins=100)
a_b = np.histogram(b_b, bins=100)
ax[2].set_xlabel('pixel level')
ax[2].set_ylabel('Frequency')
bins_loc_e = (a_e[1][0:-1] + a_e[1][1:]) / 2
bins_loc_b = (a_b[1][0:-1] + a_b[1][1:]) / 2
ax[2].semilogy(bins_loc_e, a_e[0] / sum(a_e[0]), 'b-', bins_loc_b, a_b[0] / sum(a_b[0]), 'g-')
ax[2].title.set_text('V histogram comparison of good quality images: b:Eileen;g:holdout')

ax[2].grid()
plt.savefig(os.path.join(results_path, 'HSV_hist.png'))
plt.show()



# kwargs = dict(histtype='stepfilled', alpha=0.3,  bins=200)
# ax2 = plt.figure(101)
# hh2 = ax2.gca()
# hh2.hist(data, **kwargs)
