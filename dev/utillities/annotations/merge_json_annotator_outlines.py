import os
import json
import pickle

path = '/hdd/blind_raw_images/ee40624'

filenames = [os.path.join(path, x) for x in os.listdir(path)
             if x.endswith('json')]
d_contour = dict()
for file in filenames:
    with open(file) as f:
        data_json = json.load(f)
        fname = file.split('/')[-1].split('.json')[0] + '.png'

        d_contour.update({fname: {'contour': data_json}})

with open(os.path.join(path, 'contours.pkl'), 'wb') as f:
    pickle.dump(d_contour, f)

"""

cutout_2_outline_n_meta['blind_20200712T075802.123556Z_0.png'].keys()
Out[12]: dict_keys(['quality', 'datetime', 'contour'])


Out of 144
bad/marginal 
blind_20201220T133750.749467Z_0.png
blind_20201220T122711.543931Z_0
blind_20201220T124111.680518Z_0
blind_20201220T133751.314927Z_0
blind_20201220T133751.314927Z_0
partially burned by flash

Burned
blind_20201221T094947.061714Z_0
blind_20201221T102130.605103Z_0
blind_20201221T133124.717388Z_0

blind_20201222T092728.313698Z_0 + leopard spots
blind_20201222T100824.467396Z_0

blind_20201221T102130.605103Z_0
blind on the move  : blind_20201221T090457.404570Z_0  + blind_20201221T090457.680925Z_0
blind_20201221T103404.595760Z_0
blind_20201221T133124.161822Z_0
blind_20201221T133124.717388Z_0
blind_20201222T100824.467396Z_0
blind_20201222T124114.277439Z_0
blind_20201222T124549.615401Z_0
blind_20201222T125357.688749Z_0
blind_20201223T135722.394154Z_0
blind_20201223T135749.817176Z_1
blind_20201224T112422.664877Z_0
blind_20201224T112423.218482Z_1
blind_20201224T112526.377792Z_0
blind_20201224T112802.878479Z_0
blind_20201224T121925.609733Z_1
blind_20201224T122804.430581Z_0
blind_20201226T110549.279266Z_0
blind_20201226T113027.628386Z_0
blind_20201226T123408.841267Z_1
blind_20201228T111637.516955Z_0
blind_20201228T114723.998531Z_0
blind_20201229T101246.973867Z_0
blind_20201229T101916.711503Z_0
blind_20201229T101916.711503Z_0
"""