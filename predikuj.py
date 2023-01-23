import shutil

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.dataset_conversion.Task032_BraTS_2018 import convert_labels_back_to_BraTS_2018_2019_convention


def main():
    input_folder = '/input'
    output_folder = '/output'

    # tmp_input_folder = '/tmp_input'
    # tmp_output_folder = '/tmp_output'
    # maybe_mkdir_p(tmp_input_folder)
    os.system("export RESULTS_FOLDER=/workspace/nnUNet/nnUNet_trained_models/")
    # convert raw data to nnunet format
    # contrast_to_number = {'t1': '0000', 't1ce': '0001', 't2': '0002', 'flair': '0003'}
    # for p in subfiles(input_folder, join=False):
    #     tokens = p.split('_')
    #     patient_id = tokens[0] + "_" + tokens[1]
    #     contrast = tokens[-1].split('.')[0]
    #     shutil.copy(join(input_folder, p),
    #                 join(tmp_input_folder, patient_id + "_" + contrast_to_number[contrast] + ".nii.gz"))

    # run nnunet inference
    # tmp_output_folder_inferencia = join(tmp_output_folder, 'raw_output_1')
    os.system(
        "nnUNet_predict -i {} -o {} -t 600 -m 3d_fullres -tr diplomovka_nnUNetTrainer --save_npz".format(
            input_folder, output_folder))
    # convert_labels_back_to_BraTS_2018_2019_convention(join(output_folder, 'pp_output'),
    #                                                   join(output_folder, 'pp_output_converted'))
    #
    # for p in subfiles(join(output_folder, 'pp_output_converted'), join=False):
    #     patient_id = p.split('_')[1].split('.')[0]
    #     shutil.copy(join(output_folder, 'pp_output_converted', p), join(output_folder, patient_id + ".nii.gz"))


if __name__ == '__main__':
    main()
    # print("ciao")
