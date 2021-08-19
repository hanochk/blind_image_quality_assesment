import numpy as np
from dev.dataset import class_labels
# from dataset import class_labels

def filter_low_conf_high_loss_examples_from_dataframe(all_predictions, all_targets, image_names, test_df, th=0.05):
    det_good_cls = all_predictions[:, class_labels['good']][all_targets == 1]
    det_bad_cls = all_predictions[:, class_labels['bad']][all_targets == 0] # llr of bad class that should be classified as bad
    det_good_image_names = image_names[all_targets == 1]
    det_bad_cls_image_names = image_names[all_targets == 0] # llr of bad class that should be classified as bad

    if 1:
        import numpy as np
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        a = np.histogram(det_good_cls, bins=100)
        bins_loc = (a[1][0:-1] + a[1][1:]) / 2
        ax.plot(bins_loc, a[0] / sum(a[0]))
        plt.grid()
        plt.title('Histogram of good class confidence')
        plt.show()

        plt.figure()
        cs = np.cumsum(a[0]) / a[0].sum()
        plt.plot(bins_loc, cs)
        plt.grid()
        plt.title('CDF of good class confidence')
        plt.show()

        fig, ax = plt.subplots()
        a = np.histogram(det_bad_cls, bins=100)
        bins_loc = (a[1][0:-1] + a[1][1:]) / 2
        ax.semilogy(bins_loc, a[0] / sum(a[0]))
        plt.grid()
        plt.title('Histogram of bad class confidence')
        plt.show()

        plt.figure()
        cs = np.cumsum(a[0]) / a[0].sum()
        plt.plot(bins_loc, cs)
        plt.grid()
        plt.title('CDF of bad class confidence')
        plt.show()


    det_good_cls_image_names_below_th = det_good_image_names[det_good_cls < th]
    det_good_cls_conf = det_good_cls[det_good_cls < th]
    det_bad_cls_image_names_below_th = det_bad_cls_image_names[det_bad_cls < th]
    det_bad_cls_conf = det_bad_cls[det_bad_cls < th]
    print("N good tiles filtered {} below th {}".format(det_good_cls_image_names_below_th.shape, th))
    print("N bad tiles filtered {} below th {}".format(det_bad_cls_image_names_below_th.shape, th))
    # sanity
    extract_img_names = np.where(image_names == det_good_image_names[det_good_cls < th][0])[0]
    if extract_img_names.size > 0:
        ind = extract_img_names.item()
        if all_predictions[ind, 1] > th:
            raise ValueError('Warning sanity fails')

    test_df['file_name_with_png'] = test_df['file_name'].apply(lambda x: x+ '.png')
    if 0:
        import subprocess
        for iter, _ in enumerate(det_good_cls_image_names_below_th):
            full_file = test_df[test_df['file_name_with_png'] == det_good_cls_image_names_below_th[iter]].file_name_with_png.item()
            file_full_path = subprocess.getoutput(
                'find ' + '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data' + ' -iname ' + '"*' + full_file + '"')
            # print(file_full_path)
            if '\n' in file_full_path:
                file_full_path = file_full_path.split('\n')[0]
            dest_file = 'conf_' + str(det_good_cls_conf[iter].__format__('.3f')) + '_' + full_file
            ans2 = subprocess.getoutput('cp -p ' + file_full_path + ' ' + ' ' + '/hdd/hanoch/runmodels/img_quality/tail_prob/good' + '/' + dest_file)

        for iter, _ in enumerate(det_bad_cls_image_names_below_th):
            full_file = test_df[test_df['file_name_with_png'] == det_bad_cls_image_names_below_th[iter]].file_name_with_png.item()
            file_full_path = subprocess.getoutput(
                'find ' + '/hdd/hanoch/data/objects-data-bbox-20191106-simple-sharded-part/tile_data' + ' -iname ' + '"*' + full_file + '"')
            # print(file_full_path)
            if '\n' in file_full_path:
                file_full_path = file_full_path.split('\n')[0]
            dest_file = 'conf_' + str(det_bad_cls_conf[iter].__format__('.3f')) + '_' + full_file
            ans2 = subprocess.getoutput('cp -p ' + file_full_path + ' ' + ' ' + '/hdd/hanoch/runmodels/img_quality/tail_prob/bad' + '/' + dest_file)


    # filter good class
    for iter, _ in enumerate(det_good_cls_image_names_below_th):
        index_names = test_df[test_df['file_name_with_png'] == det_good_cls_image_names_below_th[iter]].index
        test_df.drop(index_names, inplace=True)

    # filter bad class
    for iter, _ in enumerate(det_bad_cls_image_names_below_th):
        index_names = test_df[test_df['file_name_with_png'] == det_bad_cls_image_names_below_th[iter]].index
        test_df.drop(index_names, inplace=True)

    del test_df['full_file_name']
    del test_df['file_name_with_png']

    return det_good_cls_image_names_below_th, det_bad_cls_image_names_below_th, test_df