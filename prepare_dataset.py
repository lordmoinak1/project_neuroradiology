import os
import json
import pandas as pd

def rsna_miccai_radiogenomics():
    files = os.listdir('/home/moibhattacha/datasets/brats21/TrainingData')
    brats21_list = [int(i.split('_')[1]) for i in files]
    df = pd.read_csv('/home/moibhattacha/project_neuroradiology/train_labels.csv')

    count = 0
    filename = '/home/moibhattacha/project_neuroradiology/rsna_miccai_radiogenomics_cross_validation.json'
    for id, label in zip(df['BraTS21ID'], df['MGMT_value']):
        if id in brats21_list:
            # print(id, label)
            id_x = str(id).zfill(5)
            image_list = [
                "TrainingData/BraTS2021_{}/BraTS2021_{}_flair.nii.gz".format(id_x, id_x),
                "TrainingData/BraTS2021_{}/BraTS2021_{}_t1ce.nii.gz".format(id_x, id_x),
                "TrainingData/BraTS2021_{}/BraTS2021_{}_t1.nii.gz".format(id_x, id_x),
                "TrainingData/BraTS2021_{}/BraTS2021_{}_t2.nii.gz".format(id_x, id_x),
            ]
            seg_image = 'TrainingData/BraTS2021_{}/BraTS2021_{}_seg.nii.gz'.format(id_x, id_x)
            with open(filename, 'r+') as file:
                if count >= 0 and count <= 116:
                    subject = {
                        'image': image_list,
                        'label_segmentation': seg_image,
                        'label_classification': label,
                        'fold': 0
                    }
                elif count >= 117 and count <= 233:
                    subject = {
                        'image': image_list,
                        'label_segmentation': seg_image,
                        'label_classification': label,
                        'fold': 1
                    }
                elif count >= 234 and count <= 350:
                    subject = {
                        'image': image_list,
                        'label_segmentation': seg_image,
                        'label_classification': label,
                        'fold': 2
                    }
                elif count >= 351 and count <= 467:
                    subject = {
                        'image': image_list,
                        'label_segmentation': seg_image,
                        'label_classification': label,
                        'fold': 3
                    }
                else: 
                    subject = {
                        'image': image_list,
                        'label_segmentation': seg_image,
                        'label_classification': label,
                        'fold': 4
                    }

                file_data = json.load(file)
                file_data["training"].append(subject)
                file.seek(0)
                json.dump(file_data, file, indent = 4)

            count += 1

if __name__ == "__main__":
    rsna_miccai_radiogenomics()

    # import shutil

    # files = os.listdir('/home/mbhattac/datasets/brats21/TrainingData')
    # brats21_list = [int(i.split('_')[1]) for i in files]
    # df = pd.read_csv('/home/mbhattac/project_neuroradiology/train_labels.csv')

    # for id in df['BraTS21ID']:
    #     if id in brats21_list:
    #         id_x = str(id).zfill(5)
    #         shutil.copy('/home/mbhattac/project_neuroradiology/outputs/BraTS2021_{}.nii.gz'.format(id_x), '/home/mbhattac/project_neuroradiology/outputs_x/{}.nii.gz'.format(id_x))
