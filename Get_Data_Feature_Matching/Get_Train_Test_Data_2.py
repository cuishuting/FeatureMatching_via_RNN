import numpy as np


"""*_2 means only generate 2 data files: *_MV.npz and *_CV.npz"""
"""get *_MV.npz"""
total_map_mv = np.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/imputed_map_features_MV_1_24.npz")
total_unmap_mv = np.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/imputed_unmap_features_MV_1_24.npz")
# "ep_tdata"
total_mv_ep_tdata = np.concatenate((total_map_mv["ep_tdata"], total_unmap_mv["ep_tdata"]), axis=2)
test_mv_ep_tdata = total_mv_ep_tdata[:1000, :, :]
# "ep_tdata_masking"
total_mv_ep_tdata_masking = np.concatenate((total_map_mv["ep_tdata_masking"], total_unmap_mv["ep_tdata_masking"]), axis=2)
test_mv_ep_tdata_masking = total_mv_ep_tdata_masking[:1000, :, :]
# "adm_features_all"
total_mv_adm_features_all = total_map_mv["adm_features_all"]
test_mv_adm_features_all = total_map_mv["adm_features_all"][:1000, :]
# "y_icd9"
total_mv_icd9 = total_map_mv["y_icd9"]
test_mv_icd9 = total_map_mv["y_icd9"][:1000, :]
# "y_mor"
total_mv_y_mor = total_map_mv["y_mor"]
test_mv_y_mor = total_map_mv["y_mor"][:1000]

np.savez_compressed("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/imputed_MV_1_24.npz",
                    ep_tdata=total_mv_ep_tdata,
                    ep_tdata_masking=total_mv_ep_tdata_masking,
                    adm_features_all=total_mv_adm_features_all,
                    y_icd9=total_mv_icd9,
                    y_mor=total_mv_y_mor)
np.savez_compressed("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/test_imputed_MV_1_24.npz",
                    ep_tdata=test_mv_ep_tdata,
                    ep_tdata_masking=test_mv_ep_tdata_masking,
                    adm_features_all=test_mv_adm_features_all,
                    y_icd9=test_mv_icd9,
                    y_mor=test_mv_y_mor)
"""get *_CV.npz"""
total_map_cv = np.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/imputed_map_features_CV_1_24.npz")
total_unmap_cv = np.load("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/imputed_unmap_features_CV_1_24.npz")
# "ep_tdata"
total_cv_ep_tdata = np.concatenate((total_map_cv["ep_tdata"], total_unmap_cv["ep_tdata"]), axis=2)
test_cv_ep_tdata = total_cv_ep_tdata[:1000, :, :]
# "ep_tdata_masking"
total_cv_ep_tdata_masking = np.concatenate((total_map_cv["ep_tdata_masking"], total_unmap_cv["ep_tdata_masking"]), axis=2)
test_cv_ep_tdata_masking = total_cv_ep_tdata_masking[:1000, :, :]
# "adm_features_all"
total_cv_adm_features_all = total_map_cv["adm_features_all"]
test_cv_adm_features_all = total_cv_adm_features_all[:1000, :]
# "y_icd9"
total_cv_y_icd9 = total_map_cv["y_icd9"]
test_cv_y_icd9 = total_cv_y_icd9[:1000, :]
# "y_mor"
total_cv_y_mor = total_map_cv["y_mor"]
test_cv_y_mor = total_cv_y_mor[:1000]

np.savez_compressed("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/imputed_CV_1_24.npz",
                    ep_tdata=total_cv_ep_tdata,
                    ep_tdata_masking=total_cv_ep_tdata_masking,
                    adm_features_all=total_cv_adm_features_all,
                    y_icd9=total_cv_y_icd9,
                    y_mor=total_cv_y_mor)

np.savez_compressed("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/MIMIC_data_new_by_cst/test_imputed_CV_1_24.npz",
                    ep_tdata=test_cv_ep_tdata,
                    ep_tdata_masking=test_cv_ep_tdata_masking,
                    adm_features_all=test_cv_adm_features_all,
                    y_icd9=test_cv_y_icd9,
                    y_mor=test_cv_y_mor)