import unittest
from modules.blind_quality.quality import blindQualityNode
from modules.blind_quality.quality_utils import get_gpu_free_memory, get_gpu_total_memory, check_if_gpu
import json
from collections import namedtuple
from skimage import io
import numpy as np

check_if_gpu_exist = False # causing problem in double GPU invocation, GPU mem alloc doesn't comply to restriction TODO reset the GPU allocation

class MyTestCase(unittest.TestCase):

    def test_bit_exactness_throughput_mem(self):
        Point = namedtuple('Point', ['x', 'y'])
        gpu_id = 0
# Check if GPU exist this test fails if GPU is busy hence
        if check_if_gpu_exist:
            is_cuda_gpu_available = check_if_gpu()
            if is_cuda_gpu_available:
                print("CUDA GPU found! {}".format(is_cuda_gpu_available))
            self.assertTrue(is_cuda_gpu_available)

        reference_rsult_dict_path = './tests/test_data/unittest.json'
        file = './tests/test_data/unittest.png'

        with open(reference_rsult_dict_path, 'r') as f:
            reference_rsult_dict = json.load(f)

        print(reference_rsult_dict.keys())
        print(reference_rsult_dict['outline'])
        outline = [Point(val[0], val[1]) for val in reference_rsult_dict['outline']]
        print(outline)
        blind_quality = blindQualityNode(gpu_id=0,
                                       debug_is_active=True)

        print("Threshold integrity :")
        self.assertTrue(blind_quality.average_pooling_threshold_loose_for_secondary_use < blind_quality.images_soft_fusion_th) #

        # load ref vector for test

        #load image
        image = io.imread(file)
        blind_quality_category, soft_score = blind_quality.handle_image(image=image, contour=outline)
        print("Final quality : {}". format(blind_quality_category))
        # resulat_stat_acc.update({cutout_id: blind_quality.nested_record_stat})
# compare results
# compare final label
        print("Testing quality classifier label output {}".format(blind_quality_category))
        self.assertTrue(blind_quality.nested_record_stat['pred_label'] == blind_quality_category)
        #Compare confidence
        self.assertTrue(soft_score < blind_quality.images_soft_fusion_th) #
        self.assertTrue(np.isclose(soft_score.astype('float'), float(reference_rsult_dict['soft_score']), atol=1e-3))

        # compare LLRs
        llr_ref_model = reference_rsult_dict['tile_good_class_pred']
        llr_tf_model = blind_quality.nested_record_stat['tile_good_class_pred']

        self.assertTrue(len(llr_ref_model) == len(llr_tf_model))
        err = llr_tf_model - llr_ref_model
        print("Bit exactness test : sum of abs error {} cutout id {}".format(np.sum(np.abs(err)), reference_rsult_dict['cutout_id']))
        self.assertTrue(np.isclose(llr_tf_model, llr_ref_model, atol=1e-08).any())

#       Throughput test
        print("Throughput test : {} for {} tiles".format(blind_quality.model_run_time[-1], len(llr_ref_model)))
        if blind_quality.model_run_time[-1] > 1.3: #support NVIDIA 3060
            print("Throughput test failed {} > 1sec for {} tiles!!".format(blind_quality.model_run_time[-1], len(llr_ref_model)))
            self.assertEqual(True, False)
# Test GPU mem size
        total_memory = get_gpu_total_memory()
        free_mem = get_gpu_free_memory()
        occuppied_mem = np.array(total_memory) - np.array(free_mem)
        print("GPU mem test :  {} >< 2600 [MB]".format(occuppied_mem[gpu_id]))
        self.assertTrue(occuppied_mem[gpu_id] < 2600) # MB

if __name__ == '__main__':
    unittest.main()

#python -m unittest /home/hanoch/GIT/mahitl-aqua-blind-quality-cnn/modules/test_quality.py