import shutil
from collections import OrderedDict

import numpy as np
import scipy.stats as ss
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.dataset_conversion.Task043_BraTS_2019 import copy_BraTS_segmentation_and_convert_labels
from nnunet.evaluation.region_based_evaluation import get_brats_regions, evaluate_regions
from nnunet.paths import nnUNet_raw_data


def rank_algorithms(data: np.ndarray):
    """
    data is (metrics x experiments x cases)
    :param data:
    :return:
    """
    num_metrics, num_experiments, num_cases = data.shape
    ranks = np.zeros((num_metrics, num_experiments))
    for m in range(6):
        r = np.apply_along_axis(ss.rankdata, 0, -data[m], 'min')
        ranks[m] = r.mean(1)
    average_rank = np.mean(ranks, 0)
    final_ranks = ss.rankdata(average_rank, 'min')
    return final_ranks, average_rank, ranks


def load_brats_validation_csv(csv_file):
    data = np.loadtxt(csv_file, dtype='str', delimiter=',')
    data = data[1:-5, 1:7].astype(float)
    return data


def my_rank_then_aggregate_on_validation_result(result_dir):
    """
    Similar to the above code but rank on result on brats validation set
    """
    submissions = subdirs(result_dir, join=False)
    data_for_ranking = np.zeros((6, len(submissions), 219), dtype=np.float32)
    for idx, sub in enumerate(submissions):
        curr = join(result_dir, sub)
        validation_score = load_brats_validation_csv(join(curr, 'all_scores.csv'))
        data_for_ranking[:, idx, :] = validation_score.transpose()

    final_ranks, average_rank, ranks = rank_algorithms(data_for_ranking)
    print(submissions)
    print(final_ranks, average_rank, ranks)


def my_evaluate_folder(folder, gt_folder):
    # evaluate arbirtrary folder, good for checking ensemble of validation result
    regions = get_brats_regions()
    evaluate_regions(folder, gt_folder, regions)


if __name__ == "__main__":
    # task_name = "Task500_Diplomovka"
    # downloaded_data_dir = "/home/grafika/Desktop/train/"
    # downloaded_data_dir_test = "/home/grafika/Desktop/test/"
    # result_dir = "/home/grafika/Desktop/Task500_Diplomovka/"
    #
    # target_base = join(nnUNet_raw_data, task_name)
    # target_imagesTr = join(target_base, "imagesTr")
    # target_imagesTs = join(target_base, "imagesTs")
    # target_labelsTr = join(target_base, "labelsTr")
    # target_labelsTs = join(target_base, "labelsTs")
    #
    # maybe_mkdir_p(target_imagesTr)
    # maybe_mkdir_p(target_imagesTs)
    # maybe_mkdir_p(target_labelsTr)
    # maybe_mkdir_p(target_labelsTs)
    #
    # patient_names = []
    # cur = join(downloaded_data_dir)
    # for p in subdirs(cur, join=False):
    #     patdir = join(cur, p)
    #     patient_name = p
    #     patient_names.append(patient_name)
    #     t1 = join(patdir, p + "_t1.nii.gz")
    #     t1c = join(patdir, p + "_t1ce.nii.gz")
    #     t2 = join(patdir, p + "_t2.nii.gz")
    #     flair = join(patdir, p + "_flair.nii.gz")
    #     seg = join(patdir, p + "_seg.nii.gz")
    #
    #     assert all([
    #         isfile(t1),
    #         isfile(t1c),
    #         isfile(t2),
    #         isfile(flair),
    #         isfile(seg)
    #     ]), "%s" % patient_name
    #
    #     shutil.copy(t1, join(target_imagesTr, patient_name + "_0000.nii.gz"))
    #     shutil.copy(t1c, join(target_imagesTr, patient_name + "_0001.nii.gz"))
    #     shutil.copy(t2, join(target_imagesTr, patient_name + "_0002.nii.gz"))
    #     shutil.copy(flair, join(target_imagesTr, patient_name + "_0003.nii.gz"))
    #
    #     copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTr, patient_name + ".nii.gz"))
    #
    # json_dict = OrderedDict()
    # json_dict['name'] = "Diplomovka_BraTS_2022"
    # json_dict['description'] = "nothing"
    # json_dict['tensorImageSize'] = "4D"
    # json_dict['reference'] = "see BraTS2022"
    # json_dict['licence'] = "see BraTS2022 license"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "T1",
    #     "1": "T1ce",
    #     "2": "T2",
    #     "3": "FLAIR"
    # }
    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "edema",
    #     "2": "non-enhancing",
    #     "3": "enhancing",
    # }
    # json_dict['numTraining'] = len(patient_names)
    # json_dict['numTest'] = 0
    # json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
    #                          patient_names]
    # json_dict['test'] = []
    #
    # save_json(json_dict, join(target_base, "dataset.json"))
    #
    # if downloaded_data_dir_test is not None:
    #     for p in subdirs(downloaded_data_dir_test, join=False):
    #         patdir = join(downloaded_data_dir_test, p)
    #         patient_name = p
    #         t1 = join(patdir, p + "_t1.nii.gz")
    #         t1c = join(patdir, p + "_t1ce.nii.gz")
    #         t2 = join(patdir, p + "_t2.nii.gz")
    #         flair = join(patdir, p + "_flair.nii.gz")
    #         seg = join(patdir, p + "_seg.nii.gz")
    #
    #         assert all([
    #             isfile(t1),
    #             isfile(t1c),
    #             isfile(t2),
    #             isfile(flair),
    #             isfile(seg)
    #         ]), "%s" % patient_name
    #
    #         shutil.copy(t1, join(target_imagesTs, patient_name + "_0000.nii.gz"))
    #         shutil.copy(t1c, join(target_imagesTs, patient_name + "_0001.nii.gz"))
    #         shutil.copy(t2, join(target_imagesTs, patient_name + "_0002.nii.gz"))
    #         shutil.copy(flair, join(target_imagesTs, patient_name + "_0003.nii.gz"))
    #
    #         copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTs, patient_name + ".nii.gz"))

    my_evaluate_folder('/home/grafika/Desktop/prediction_ag_8heads',
                       '/home/grafika/Pictures/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Diplomovka/labelsTs')

# my_rank_then_aggregate_on_validation_result
