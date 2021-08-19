from src.CNN.test_blind_quality import plot_image_and_outline
from PIL import Image
import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import glob
import pickle
from collections import namedtuple
import pandas as pd

Point = namedtuple('Point', ['x', 'y'])
def warp_cyclic_series_to_continuous(sobjectsd_outline_cand, len_outline):
    ind_series = [x[0] for x in sobjectsd_outline_cand]
    # in case 0 inside the series there is a chance that [0, 1, 2, 3, 4, 47, 48, 49] are consecutive since 0 comes after 49
    if 0 in [x[0] for x in sobjectsd_outline_cand]:
        pivot = np.where(np.diff([x[0] for x in sobjectsd_outline_cand]) != 1)[0] + 1
        if len(pivot) ==1: # more than 1 non continuous
            # print(pivot)
            # print([x[0] for x in sobjectsd_outline_cand])
            return np.append(np.array(ind_series[pivot.item():]) - len_outline, np.array(ind_series[: pivot.item() - 1]))
    return np.array(ind_series)

def main(args: list = None):
    parser = ArgumentParser()

    parser.add_argument("--database-root", type=str, required=True, metavar='PATH',
                                        help="path to the csv/pkl files")

    parser.add_argument("--result-dir", type=str, required=True, metavar='PATH',
                                        help="path to the out files")

    parser.add_argument("--outline-pickle-path", type=str, required=True, default=None, metavar='PATH',
                        help="outline-pickle-path")

    parser.add_argument("--outline-file-format", type=str, required=False, default='media_id_cutout_id',
                        choices=['media_id_cutout_id', 'misc', 'other'], metavar='STRING',
                        help="")

    parser.add_argument("--process-folder-non-recursively", action='store_true', help="If true.")

    # parser.add_argument("--plot-dot-overlay-instead-of-box", action='store_true', help="If true.")

    # parser.add_argument("--media-id-info-pickle-file", type=str, required=False, default='None', metavar='PATH',
    #                                     help="media id -pickle-file path structure : the value of the cutout dict key")
    # --outline-pickle-path /hdd/blind_raw_images/ee40624/merged_outlines_json.pkl
    # args.outline_pickle_path = '/hdd/blind_raw_images/ee40624/merged_outlines_json.pkl'

    args = parser.parse_args(args)
    print(vars(args))

    df_results = pd.DataFrame(columns=['cutout_id', 'partial'])
    try:
        with open(args.outline_pickle_path, 'rb') as f:
            cutout_2_outline_n_meta = pickle.load(f)
            print('direct')
    except:
        import sys
        sys.path.append('/home/hanoch/GIT')
        sys.path.append('/home/hanoch/GIT/Finders')
        sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        from Finders.finders.blind import default as blindfinder
        with open(args.outline_pickle_path, 'rb') as f:
            cutout_2_outline_n_meta = pickle.load(f)
            print('via_Finders')


    outline_file_format = args.outline_file_format

    if args.process_folder_non_recursively:
        filenames = [os.path.join(args.database_root, x) for x in os.listdir(args.database_root)
                     if x.endswith('png')]
        print('Process folder NON - recursively')
    else:
        filenames = glob.glob(args.database_root + '/**/*.png', recursive=True)
        print('Process folder recursively')

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    for idx, file in enumerate(tqdm.tqdm(filenames)):
        if file.split('.')[-1] == 'png':
            img = Image.open(file).convert("RGB")
            image = np.asarray(img)

            quality_svm = None
            if outline_file_format is 'media_id_cutout_id':
                cutout_id = file.split('.')[-2].split('_')[-1]
                outline = cutout_2_outline_n_meta.get(cutout_id, None)
                if outline == None:
                    print("Outline for cutout_id {} was not found {}!!!".format(cutout_id, file))
                    continue

                if isinstance(outline, dict):
                    outline = outline['outline']

            else:
                cutout_id = file.split('/')[-1]
                outline_dict = cutout_2_outline_n_meta.get(cutout_id, None)
                if outline_dict == None:
                    print("Outline for cutout_id {} was not found!!!".format(cutout_id))
                    continue
                quality_svm_reported = outline_dict.get('quality', None)
                # quality_svm_reported = clsss_quality_label_2_str[outline_dict['quality']]

                outline = outline_dict.get('contour', None)
                if outline is not None:
                    if isinstance(outline, dict):
                        # outline = outline['outline']
                        if 'contour' in outline.keys():
                            outline = outline_dict['contour']
                        elif 'outline' in outline.keys():
                            outline = outline['outline']

                # outline = outline_dict.get('outline', None)
                # if outline is not None:
                # # outline = outline_dict['contour']
                #     if isinstance(outline, dict):
                #         outline = outline['outline']

                if outline == None:
                    print("Strange !!! the nested dict : Outline for cutout_id {} was not found!!!".format(cutout_id))
                    continue
            # TODO: the sobjectsd partial blind can be away from the image edges due to the blind-finder error =>
            # the min_sobjectsd_blind_th_y=func(gap_annot) = A*gap_annot + b
            # in order not to rule out blind tail that is objects sobjectsd blind away from image edges but tiny
            # for partial on x axis first sort the point according to x to avoid 2 sobjectsd parts by eval discontunuity in the : for idx, p in enumerate(outline):
            partial_flag = False
            # plot_image_and_outline(image, [outline], 10)
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.plot([p[0] for p in outline], [p[1] for p in outline], 'b-')
            #check if within strip >min_sobjectsd_blind_th the other axis within gap_annot from image edges
            # assumption: x,y tuples are ordered clockwise or counterclockwise
            gap_annot = 150 #350  # 350 over x and 150 over y is better to deal with un-tight outline
            min_sobjectsd_blind_th_x = 2000
            min_sobjectsd_blind_th_y = 500 # due to different AR for images 1:5
#partial blind on the right over y axis
            max_x = -1
            sobjectsd_outline_cand_y = list()
            sobjectsd_outline_cand = list()
            for idx, p in enumerate(outline):
                max_x = max(max_x, p.x)
                if p.x > image.shape[1] - gap_annot:
                    sobjectsd_outline_cand_y.append(p.y)
                    sobjectsd_outline_cand.append([idx, p])
                    print(p)
            if len(sobjectsd_outline_cand_y)>1 and not ((np.diff(warp_cyclic_series_to_continuous(sobjectsd_outline_cand, len(outline)))!=1).any()):
                sobjectsd_outline_cand = [x[1] for x in sobjectsd_outline_cand] # take the pint only
                sobjectsd_y_strip_len = np.max(sobjectsd_outline_cand_y) - np.min(sobjectsd_outline_cand_y)
                if sobjectsd_y_strip_len > min_sobjectsd_blind_th_y:
                    print("we have partial blind over y axis{}".format([np.min(sobjectsd_outline_cand_y),np.max(sobjectsd_outline_cand_y) ]))
                    if sobjectsd_outline_cand:
                        ax.plot([p[0] for p in sobjectsd_outline_cand], [p[1] for p in sobjectsd_outline_cand], 'r-')
                        partial_flag = True
            # partial blind on the left over y axis
            max_x = -1
            sobjectsd_outline_cand_y = list()
            sobjectsd_outline_cand = list()
            for idx, p in enumerate(outline):
                max_x = max(max_x, p.x)
                if p.x < gap_annot:
                    sobjectsd_outline_cand_y.append(p.y)
                    sobjectsd_outline_cand.append([idx, p])
                    print(p)
            if len(sobjectsd_outline_cand_y)>1 and not ((np.diff(warp_cyclic_series_to_continuous(sobjectsd_outline_cand, len(outline)))!=1).any()):
                sobjectsd_outline_cand = [x[1] for x in sobjectsd_outline_cand] # take the pint only
                sobjectsd_y_strip_len = np.max(sobjectsd_outline_cand_y) - np.min(sobjectsd_outline_cand_y)
                if sobjectsd_y_strip_len > min_sobjectsd_blind_th_y:
                    print("we have partial blind over y axis{}".format(
                        [np.min(sobjectsd_outline_cand_y), np.max(sobjectsd_outline_cand_y)]))
                    if sobjectsd_outline_cand:
                        ax.plot([p[0] for p in sobjectsd_outline_cand], [p[1] for p in sobjectsd_outline_cand],
                                            'r-')
                        partial_flag = True

#partial blind sobjectsd on lower x axis
            max_y = -1
            sobjectsd_outline_cand_x = list()
            sobjectsd_outline_cand = list()
            for idx, p in enumerate(outline):
                max_y = max(max_y, p.y)
                if p.y > image.shape[0] - gap_annot:
                    sobjectsd_outline_cand_x.append(p.x)
                    sobjectsd_outline_cand.append([idx, p])
                    print(p)
#check consecutive points that fullfill the partial blind cond.
            if len(sobjectsd_outline_cand_x)>1 and not ((np.diff(warp_cyclic_series_to_continuous(sobjectsd_outline_cand, len(outline)))!=1).any()):
                sobjectsd_outline_cand = [x[1] for x in sobjectsd_outline_cand] # take the pint only
                sobjectsd_x_strip_len = np.max(sobjectsd_outline_cand_x) - np.min(sobjectsd_outline_cand_x)
                if sobjectsd_x_strip_len > min_sobjectsd_blind_th_x:
                    print("we have partial blind over x axis{}".format([np.min(sobjectsd_outline_cand_x),np.max(sobjectsd_outline_cand_x) ]))
                    if sobjectsd_outline_cand:
                        ax.plot([p[0] for p in sobjectsd_outline_cand], [p[1] for p in sobjectsd_outline_cand], 'r-')
                        partial_flag = True

            # partial blind sobjectsd on upper x axis
            max_y = -1
            sobjectsd_outline_cand_x = list()
            sobjectsd_outline_cand = list()
            for idx, p in enumerate(outline):
                max_y = max(max_y, p.y)
                if p.y < gap_annot:
                    sobjectsd_outline_cand_x.append(p.x)
                    sobjectsd_outline_cand.append([idx, p])
                    print(p)
            if len(sobjectsd_outline_cand_x)>1 and not ((np.diff(warp_cyclic_series_to_continuous(sobjectsd_outline_cand, len(outline)))!=1).any()):
                sobjectsd_outline_cand = [x[1] for x in sobjectsd_outline_cand] # take the pint only
                sobjectsd_x_strip_len = np.max(sobjectsd_outline_cand_x) - np.min(sobjectsd_outline_cand_x)
                if sobjectsd_x_strip_len > min_sobjectsd_blind_th_x:
                    print("we have partial blind over x axis{}".format([np.min(sobjectsd_outline_cand_x),np.max(sobjectsd_outline_cand_x) ]))
                    if sobjectsd_outline_cand:
                        ax.plot([p[0] for p in sobjectsd_outline_cand], [p[1] for p in sobjectsd_outline_cand], 'r-')
                        partial_flag = True

            tag = ['partial_detected' if partial_flag is True else '']
            plt.savefig(os.path.join(args.result_dir, 'CContour_over_the_blind_' + str(cutout_id) + '_' + tag[0] + '.png'))
            df_results.loc[len(df_results)] = [cutout_id, partial_flag]

    df_results.to_csv(os.path.join(args.result_dir, 'results'), index=False)
if __name__ == "__main__":
    main()
