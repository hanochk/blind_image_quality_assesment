import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly

path = '/hdd/hanoch/data/database/png_blind_q_2nd_batch_2529'
file = 'query_with_quality1_1900annotator_hist.csv'
df = pd.read_csv(os.path.join(path, file), index_col=False)
quality_to_labels_annotator_version = {'poor': 1, 'marginal': 2, 'excellent': 3}
df = df.replace(np.nan, '', regex=True)

ex_score_vs_eb = list()
ex_score_vs_eb_plus_1anno = list()
ex_score_vs_eb_plus_2anno = list()
ex_score_vs_eb_plus_3anno = list()
ex_score_vs_eb_plus_4anno = list()
ex_score_vs_eb_plus_5anno = list()
ex_score_vs_eb_plus_6anno = list()
ex_score_vs_eb_plus_7anno = list()

for ind, row in df.iterrows():
    if row.eb != '':
        eb_q = quality_to_labels_annotator_version[row.eb]
        example_score = list()
        for ele in row.keys():
            if ele != 'eb': # eb the supervisor already considered
                score_anno = row[ele]
                if score_anno != '':
                    # print(score_anno)
                    example_score.append(quality_to_labels_annotator_version[score_anno])
        n_annotator = len(example_score) # besides eb
        if n_annotator>0:
            example_score_arr = np.array(example_score).mean()
            globals()[f'ex_score_vs_eb_plus_{n_annotator}anno'].append([eb_q, example_score_arr])
            ex_score_vs_eb.append([eb_q, example_score_arr])

diff_mat = [it[1]-it[0] for it in ex_score_vs_eb]
a_b = np.histogram(diff_mat, bins=100, density=True)
bins_loc_b = (a_b[1][0:-1] + a_b[1][1:]) / 2
# fig, ax = plt.subplots()
# # ax.semilogy(bins_loc_b, a_b[0] / sum(a_b[0]))
# ax.plot(bins_loc_b, 100*a_b[0] / sum(a_b[0]), '')
# plt.xlabel('Average of annotators - supervisor decision')
# plt.ylabel('Percentage')
# plt.grid()
# plt.show()


fig, ax = plt.subplots()
# ax.semilogy(bins_loc_b, a_b[0] / sum(a_b[0]))
ax.bar(bins_loc_b, 100*(a_b[0] / sum(a_b[0])), width=0.05)
plt.xlabel('Average of annotators - supervisor decision')
plt.ylabel('Percentage')
plt.grid()


fig, ax = plt.subplots()
plt.grid()
plt.hist(diff_mat, bins=100)
plt.xlabel('Average of annotators - supervisor decision')

ex_score_vs_eb_arr = np.array(ex_score_vs_eb)
for lbl in np.unique(ex_score_vs_eb_arr[:, 0]):
    lbl_ind = np.where(ex_score_vs_eb_arr[:, 0] == lbl)
    avg_annot_score_given_gt = ex_score_vs_eb_arr[lbl_ind, 1]

    a_b = np.histogram(avg_annot_score_given_gt, bins=100, density=True)
    bins_loc_b = (a_b[1][0:-1] + a_b[1][1:]) / 2

    fig, ax = plt.subplots()
    # ax.semilogy(bins_loc_b, a_b[0] / sum(a_b[0]))
    ax.bar(bins_loc_b, 100*(a_b[0] / sum(a_b[0])), width=0.05)
    plt.title('Average of annotators given supervisor decision is {}'.format(lbl))
    plt.ylabel('Percentage')
    plt.grid()
for label in np.unique(ex_score_vs_eb_arr[:, 0]):
    print("Label {} support {:d}".format(label, np.array(ex_score_vs_eb_arr[:, 0] == label).sum()))
plt.show()
