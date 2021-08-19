import numpy as np
import re

dl = np.load('/home/hanoch/GIT/blind_quality_svm/bin/trn/template_parameter_file/fixed_C_200_prod_c9ed454e2e01de86143e9b60129030719f3e9bd8_2020_05_04_15_05_43_UTC___deep.pickle', allow_pickle=True)
dl['trained_rounds'][0].blind_svm_params.model_handlers['bad-vs-good'].clf
dl['trained_rounds'][0].blind_svm_params.model_handlers['bad-vs-good'].model_dict
dl['trained_rounds'][0].blind_svm_params.model_handlers['bad-vs-good'].restore_model()
X = np.random.rand(4, 500)
# dl['trained_rounds'][0].blind_svm_params.model_handlers['bad-vs-good'].clf.fit()
dl['trained_rounds'][0].blind_svm_params.model_handlers['bad-vs-good'].clf.predict(X.repeat(4, axis=0).transpose())
dl['trained_rounds'][0].blind_svm_params.model_handlers['bad-vs-marginal'].clf.predict(X.repeat(4, axis=0).transpose())
dl['trained_rounds'][0].blind_svm_params.model_handlers['marginal-vs-good'].clf.predict(X.repeat(4, axis=0).transpose())
# dl['trained_rounds'][-1].blind_svm_params.model_handlers['bad-vs-good'].clf.predict(X.repeat(4, axis=1))

svm_facsimile_015_my = np.load('/home/hanoch/GIT/blind_quality_svm/bin/trn/svm_benchmark/FACSIMILE_PRODUCTION_V0_1_5/fixed_C_200_facsimile_hk___deep.pickle', allow_pickle=True)






