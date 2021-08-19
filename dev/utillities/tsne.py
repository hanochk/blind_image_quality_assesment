import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import numpy as np
import pickle
n_components = 2
perplexity = 30


# Before normalization
# self.blind_svm_params.predictors_raw_training
# self.blind_svm_params.judgments_training
# tsne_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(self.blind_svm_params.predictors_raw_training)
# plt.title("TSNE of classes un_normed {} N={} p={}".format(np.unique(self.blind_svm_params.judgments_training), self.blind_svm_params.judgments_training.shape[0], perplexity))
# plt.savefig(os.path.join('/hdd/hanoch/results','tsne_un_normed' 'p_' + str(perplexity)+ '.png'))

# mapping unscaled to scaled
# inv = self.scaler.inverse_transform(self.predictors_normalized_training)
#  Mapped to unscaled
# self.blind_svm_params.predictors_raw_training[0]
def plot_tsne(all_targets, all_features, path, fname='tsne_' 'p_', perplexity=30, n_components = 2):
    from sklearn.manifold import TSNE
    fig = plt.figure()

    tsne_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(all_features)
    tsne_dict = {'tsne_data': tsne_data, 'all_targets': all_targets, 'perplexity': perplexity}

    with open(os.path.join(path, fname + 'tsne_vec_' + str(perplexity)), 'wb') as fh:
        pickle.dump(tsne_dict, fh)

    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=all_targets, s=1)
    # plt.scatter(tsne_data[:, 0][all_targets == 2], tsne_data[:, 1][all_targets == 2], c='r', s=10, marker='*')
    plt.title("TSNE of classes {} N={} p={}".format(np.unique(all_targets), all_targets.shape[0], perplexity))
    plt.savefig(os.path.join(path, fname + str(perplexity) + '.png'))

def main():
    path = '/hdd/hanoch/results/tsne_domain_shift_cage'
    mat_pkl = 'train_test_val_quality_tile_filtered_eileen_good_bad_val_cnn_features.pkl'
    perplexity = 50
    print("tSNE with perplexity {}".format(perplexity))
    with open(os.path.join(path, mat_pkl), 'rb') as fh:
        mat_data = pickle.load(fh)

    cage_pkl = 'blind_low_conf_cnn_features.pkl'

    with open(os.path.join(path, cage_pkl), 'rb') as fh:
        cage_shifted_data = pickle.load(fh)

    #change label for OOD
    cage_shifted_data['all_targets'] = 2 * np.ones_like(cage_shifted_data['all_targets'])
    targets = np.append(cage_shifted_data['all_targets'][:, np.newaxis], mat_data['all_targets'][:, np.newaxis])
    features = np.concatenate((cage_shifted_data['all_features'], mat_data['all_features']), axis=0)
    fname = 'merged' + 'tsne_' 'p_'
    plot_tsne(targets, features, path, fname, perplexity)

    plot_tsne(cage_shifted_data['all_targets'], cage_shifted_data['all_features'], path, '', perplexity)
    return

if __name__ == '__main__':
    main()
"""
if 0:
    tsne_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(training_data['X'])

    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=training_data['y'], s=1)
    plt.title("TSNE of classes {} N={} p={}".format(np.unique(training_data['y']), training_data['y'].shape[0], perplexity))

    # plt.show(block=True)
    plt.savefig(os.path.join('/hdd/hanoch/results','tsne_' 'p_' + str(perplexity)+ '.png'))


    feat = training_data['X']
    label = training_data['y']

    # feat = np.concatenate((feat, training_data['X']), axis=0)
    # label = np.concatenate((label, training_data['y']), axis=0)

    feat = np.concatenate((feat, training_data['X'][training_data['y']==2]), axis=0)
    label = np.concatenate((label, training_data['y'][training_data['y']==2]), axis=0)


    perplexity = 30
    # perplexity = 5
    tsne_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(feat)

    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=label, s=1)
    plt.title("TSNE of classes {} N={} p={}".format(np.unique(label), label.shape[0], perplexity))
    plt.savefig(os.path.join('/hdd/hanoch/results','tsne_all_classes_' 'p_' + str(perplexity)+ '.png'))

    # plt.show(block=True)


    # find point in specific place
    g = np.where(np.logical_and(tsne_data>[57, -30], tsne_data<=[60, -30]))[0]
    label[g][np.where(label[g]==1)]
    tsne_data[g]

    g[np.where(label[g]==1)]
    g[np.where(label[g]==3)]

    ind3 = g[np.where(label[g]==3)]
    (tsne_data[ind3], label[ind3])

    ind1 = g[np.where(label[g]==1)]
    (tsne_data[ind1], label[ind1])

    feat[ind1]
    feat[ind3]
    self.scaler.inverse_transform(feat[ind1])

    vec_lbl_1_in3 = (tsne_data[g][np.where(label[g]==1)], label[g][np.where(label[g]==1)])
    vec_lbl_3 = (tsne_data[g][np.where(label[g]==3)], label[g][np.where(label[g]==3)])

    ind1[0]
    (tsne_data[ind1[0]], label[ind1[0]])
    self.scaler.inverse_transform(feat[78,:])
    array([8.96849000e+06, 8.53171072e-02, 1.16935086e-03, 4.96892394e-02,
           1.04000409e-03, 2.77646571e-04, 5.91626332e-06, 8.23683678e-02,
           3.76542065e-03, 1.63778985e-01, 4.17819906e-03, 1.58536310e-01,
           2.94409751e-03, 4.24065282e+01, 6.23054319e+02, 5.51503578e+00,
           1.84839131e+01])
    8968490
    8968490

    self.blind_svm_params.all_parameters.keys()

    self.blind_svm_params.all_parameters['blind_keys_training'][ind1[0]]
    'ee2cb9b5-6f2e-5007-9ee7-473fb0c9017d'


    (tsne_data[88], label[88])
    (array([-58.501583 ,   6.4511614], dtype=float32), 1)

    self.scaler.inverse_transform(feat[88,:])
    array([7.79748000e+06, 7.32277537e-02, 9.33160624e-04, 4.16484784e-02,
           7.08855715e-04, 1.19305346e-02, 3.39031797e-03, 9.50370084e-02,
           1.64504355e-02, 1.64124233e-01, 1.39767861e-02, 1.47828352e-01,
           9.54947611e-03, 5.14457831e+01, 5.54102482e+02, 7.43138320e+00,
           1.87675533e+02])
    but :
    item = [ ind for ind, k in enumerate(self.blind_svm_params.all_parameters['predictors_raw_training']) if k[0]==7797480]
    item
    [133]
    self.blind_svm_params.all_parameters['blind_keys_training'][133]
    'aba8a650-6049-5faf-a917-4616e624d413'



    (tsne_data[88], label[88])
    (array([-58.501583 ,   6.4511614], dtype=float32), 1)



    (tsne_data[ind1[5:7]], label[ind1[5:7]])
    (array([[-57.292103, -12.890371],
           [-50.772503, -42.96554 ]], dtype=float32), array([1, 1]))
    label[ind1[5:7]]
    array([1, 1])

    ind1[5:7]
    array([131, 136])

    self.scaler.inverse_transform(feat[131,:])
    array([9.13688600e+06, 6.82407354e-02, 9.83392101e-04, 3.81361637e-02,
           7.35369928e-04, 2.38191239e-03, 2.76644176e-04, 7.36519378e-02,
           6.65291305e-03, 1.36776092e-01, 6.01497059e-03, 1.11631112e-01,
           3.30827680e-03, 4.31818182e+01, 7.08724518e+02, 5.87378172e+00,
           7.02648610e+01])

    item = [ ind for ind, k in enumerate(self.blind_svm_params.all_parameters['predictors_raw_training']) if k[0]==9136886]
    item
    [189]
    self.blind_svm_params.all_parameters['blind_keys_training'][189]
    'b2ac3664-eac2-5d23-8d4e-60e78d1b14c9'

    self.scaler.inverse_transform(feat[136,:])
    array([8.25523400e+06, 5.86285764e-02, 7.38194001e-04, 2.92890461e-02,
           3.76987028e-04, 7.94542101e-04, 4.88885563e-05, 4.55259640e-02,
           1.53736044e-03, 1.11018600e-01, 2.60166557e-03, 9.79583833e-02,
           2.41577778e-03, 5.06044444e+01, 6.11830202e+02, 3.17201575e+00,
           9.56608395e+00])

    item = [ ind for ind, k in enumerate(self.blind_svm_params.all_parameters['predictors_raw_training']) if k[0]==8255234]
    item
    [197]

    self.blind_svm_params.all_parameters['blind_keys_training'][197]
    '9c60670a-f2ff-5816-8189-296925da6fbc'

    # Saving the objects:
    import pickle
    with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([tsne_data, label, feat], f)

    # Getting back the objects:
    with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
        obj0, obj1, obj2 = pickle.load(f)

    import pickle
    with open('/hdd/hanoch/results/tsne.pkl','wb') as f:
        pickle.dump([ind1, feat, label, tsne_data], f)
        pickle.dump(feat, f)
        pickle.dump(label, f)
        pickle.dump(tsne_data, f)

    import pickle
    with open('/hdd/hanoch/results/tsne.pkl','rb') as f:
        while True:
            pickle.load(f)
            
"""
