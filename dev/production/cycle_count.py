import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import tqdm


path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Projects\Results\quality_eval\cycle_count\cycle_count_2'
pkl_file = '1596521630_cycle_count_production_holdout_results.pkl'

path = r'C:\Users\hanoch.kremer\OneDrive - ALLFLEX EUROPE\HanochWorkSpace\Projects\Results\quality_eval\cycle_count\cycle_count_flat_inf'
pkl_file = '1593440243_cycle_count_production_holdout_results.pkl'

model_name = pkl_file.split('_')[0]

with open(os.path.join(path, pkl_file), 'rb') as f:
    results_meta = pickle.load(f)


model_run_time_mu_per_tile_acm = []
model_run_time_std_acm = []
n_tiles_acm = []
tiles_calc_time_mu_acm = []
tiles_calc_time_std_acm = []
tiles_calc_time_mu_per_tile_acm = []
model_run_time_mu_acm = []
transform_time_mu_acm = []
cpu_gpu_time_mu_acm = []
inference_time_mu_acm = []

for cu_id in tqdm.tqdm(results_meta.keys()):
    res = results_meta[cu_id]
    n_tiles = res['n_tiles']
    tiles_calc_time_mu = res['tiles_calc_time_mu']
    tiles_calc_time_std = res['tiles_calc_time_std']
    dataset_class_prepare_time_mu = res['dataset_class_prepare_time_mu']
    dataset_class_prepare_time_std = res['dataset_class_prepare_time_std']
    dataloader_time_mu = res['dataloader_time_mu']
    dataloader_time_std = res['dataloader_time_std']
    model_run_time_mu = res['model_run_time_mu']
    model_run_time_mu_per_tile = model_run_time_mu/n_tiles
    model_run_time_mu_acm += [model_run_time_mu]
    model_run_time_std = res['model_run_time_std']
    tiles_calc_time_mu_per_tile_acm += [res['tiles_calc_time_mu']/n_tiles]

    transform_time_mu_acm += [res['transform_time_mu']]
    cpu_gpu_time_mu_acm += [res['cpu_gpu_time_mu']]
    inference_time_mu_acm += [res['inference_time_mu']]

    model_run_time_std_acm += [model_run_time_std]
    model_run_time_mu_per_tile_acm += [model_run_time_mu_per_tile]
    n_tiles_acm += [n_tiles]
    tiles_calc_time_mu_acm += [tiles_calc_time_mu]
    tiles_calc_time_std_acm += [tiles_calc_time_std]


# for ax, prcnt in enumerate(range(4)):
# filename = 'prec_recall_voting_percentile_' + str(prcnt) + model_name + 'percent.png'
avg_tiles = np.mean(n_tiles_acm)
std_tiles = np.std(n_tiles_acm)
print("average/std No. of tiles. {} : {}".format(avg_tiles, std_tiles))
fig = plt.figure()
plt.hist(model_run_time_mu_per_tile_acm, bins=100)
plt.title('Histogram of model run time per tile avg No. of tiles {}'.format(avg_tiles.__format__('.2f')))
plt.savefig(os.path.join(path, 'Histogram_of-model-run-time-per-tile.png'), format="png")

fig = plt.figure()
plt.hist(model_run_time_mu_acm, bins=100)
plt.title('Histogram of total model run time avg tiles no {} std {}'.format(avg_tiles.__format__('.2f'), std_tiles.__format__('.2f')))
plt.savefig(os.path.join(path, 'Histogram_of model_run_time_mu_acm'), format="png")

fig = plt.figure()
plt.hist(transform_time_mu_acm, bins=100)
plt.title('Histogram of transform run time avg tiles no {} std {}'.format(avg_tiles.__format__('.2f'), std_tiles.__format__('.2f')))
plt.savefig(os.path.join(path, 'Histogram_of transform run_time_mu_acm'), format="png")

fig = plt.figure()
plt.hist(cpu_gpu_time_mu_acm, bins=100)
plt.title('Histogram of cpu_gpu tx run time avg tiles no {} std {}'.format(avg_tiles.__format__('.2f'), std_tiles.__format__('.2f')))
plt.savefig(os.path.join(path, 'Histogram_of cpu_gpu tx run_time_mu_acm'), format="png")

fig = plt.figure()
plt.hist(inference_time_mu_acm, bins=100)
plt.title('Histogram of inference [GPU] time  run time avg tiles no {} std {}'.format(avg_tiles.__format__('.2f'), std_tiles.__format__('.2f')))
plt.savefig(os.path.join(path, 'Histogram_of inference_time run_time_mu_acm'), format="png")



fig = plt.figure()
plt.hist(np.array(model_run_time_mu_per_tile_acm)*int(avg_tiles), bins=100)
plt.title('Histogram of model run time with avg No. of tiles {}'.format(int(avg_tiles)))
plt.savefig(os.path.join(path, 'Histogram of model run time with avg No. of tiles.png'), format="png")

fig = plt.figure()
plt.hist(model_run_time_std_acm, bins=100)
plt.title('Histogram of model_run_time_std[sec]')
plt.savefig(os.path.join(path, 'Histogram_of-model-run-time-std.png'), format="png")

fig = plt.figure()
plt.hist(tiles_calc_time_mu_acm, bins=100)
plt.title('Histogram of tiles_calc_time_mu_acm[sec]')
plt.savefig(os.path.join(path, 'Histogram_tiles_calc_time_mu_acm.png'), format="png")

fig = plt.figure()
plt.hist(tiles_calc_time_std_acm, bins=100)
plt.title('Histogram of tiles_calc_time_std_acm[sec]')
plt.savefig(os.path.join(path, 'Histogram_tiles_calc_time_std_acm.png'), format="png")

fig = plt.figure()
plt.hist(tiles_calc_time_mu_per_tile_acm, bins=100)
plt.title('Histogram of tiles_calc_time_mu_per_tile_acm[sec]')
plt.savefig(os.path.join(path, 'Histogram_avg_tiles_calc_time_per_tile_acm.png'), format="png")


plt.ion()
plt.show()
print('ka')

# plot_thresholds_over_plt(thr, thresholds_every=10, precision_vec=np.fliplr(precision[ax, :].reshape(-1, 1)), recall_vec=np.fliplr(recall[ax, :].reshape(-1, 1)))
    # plt.grid()
    # plt.title("Class good vs. bad @ Voting percentile {} model {}".format(prcnt, model_name))
    # plt.xlabel('recall[good class]')
    # plt.ylabel('precision[good class]')
    # plt.plot(df_svm.recall, df_svm.precision, '*r')
    # plt.savefig(os.path.join(path, filename), format="png")
    # np.concatenate([np.flipud(recall[ax, :].reshape(-1, 1)), np.flipud(precision[ax, :].reshape(-1, 1))], axis=1)
