'''
Label: Huaxi no_HCM 3 level
Date: 2020/8/4
Env: test_py3
By: Haixia
'''


import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import radiomics
from radiomics import featureextractor
from matplotlib import image


class Preproces():
    def __init__(self, path_img, param_path, jpg_save_path, img_save_path, radiomic_result_path, name_path):
        self.path_img = path_img
        self.param_path = param_path
        self.jpg_save_path = jpg_save_path
        self.img_save_path = img_save_path
        self.radiomic_result_path = radiomic_result_path
        self.name_path = name_path
        # 特征提取器的设置
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(self.param_path)
        self.extractor.enableAllFeatures()
        self.extractor.enableAllImageTypes()

    # 得到文件夹下所有的子文件路径的列表
    @staticmethod
    def get_file(path):
        files = []
        if not os.path.exists(path):
            return -1
        for filepath, dirs, names in os.walk(path):
            for filename in names:
                files.append(os.path.join(filepath, filename))
        return files

    # 标准化
    # 得到文件夹下所有的子文件路径的列表
    @staticmethod
    def std_adjust(img):
        mean_img = np.mean(img)
        std_img = np.std(img)
        img_stander = (img-mean_img)/std_img
        return img_stander

    # 分开原始图片和感兴趣区域文件，这里只放进后缀为“1.nii”和“1.dcm”的文件
    @staticmethod
    def devide_img_roi(files, num):
        # print(len(files), files)
        roi_file = [i for i in files if i.endswith(num)]
        dcm_file = [i.replace(".nii", ".dcm") for i in roi_file]
        dcm_file_in = []
        roi_file_in = []
        for f in range(len(dcm_file)):
            if os.path.exists(dcm_file[f]):
                dcm_file_in.append(dcm_file[f])
                roi_file_in.append(roi_file[f])
        print('len(dcm_file), len(roi_file)', len(dcm_file), len(roi_file))
        print('dcm_file_in', dcm_file_in)
        print('files', files)
        files_p = [i.split('\\')[-2] for i in dcm_file_in]
        files_p_all = [i.split('\\')[-2] for i in files]
        files_all = []
        for p in files_p_all:
            if p not in files_all:
                files_all.append(p)
        print('len(files_p), len(files_all)', len(files_p), len(files_all))
        miss_patient = [i for i in files_all if i not in files_p]
        print('miss_patient', miss_patient)
        return dcm_file_in, roi_file_in, miss_patient

    # 是否保存感兴趣区域对应图片区域的文件（可以看感兴趣区域是否一一对应）
    @staticmethod
    def save_roi_img(img_t, img_r, name_p, save_path='./roi_img_4'):
        img_roi = img_t * img_r
        img_s = np.concatenate((img_t, img_roi, img_r * np.max(img_t)), axis=1)
        roi_path = os.path.join(save_path, name_p)
        image.imsave(roi_path, img_s)
        # plt.imshow(img_s, cmap='gray')
        # plt.show()

    # 把得到的特征存入一个dataframe中，删除了无关的“diagnostics...”信息
    @staticmethod
    def result_to_table(result_all, name_all):
        print('len(result_all)', len(result_all))
        df_0 = pd.DataFrame()
        for i in range(len(result_all)):
            result = result_all[i]
            name_p = name_all[i]
            dict_result = {}
            list_result = []
            for k, v in result.items():
                if k.startswith("diagnostics"):
                    pass
                else:
                    dict_result.update({k: v})
            # print(dict_result)
            list_result.append(dict_result)
            if i == 0:
                df_0 = pd.DataFrame(list_result)
                df_0.index = [name_p]
            else:
                df_1 = pd.DataFrame(list_result)
                df_1.index = [name_p]
                df_0 = pd.concat([df_0, df_1], axis=0)
        return df_0

    @staticmethod
    def crop_img(ori_img, roi_img):
        print(ori_img.dtype)
        crop_img_t = np.zeros((80, 80), dtype=ori_img.dtype)
        crop_img_r = np.zeros((80, 80), dtype=ori_img.dtype)
        locs = np.where(roi_img > 0)
        x_min = np.min(locs[0])
        x_max = np.max(locs[0])
        y_min = np.min(locs[1])
        y_max = np.max(locs[1])
        crop_img_t[int(40 - (x_max - x_min) / 2) - 5:int(40 + (x_max - x_min) / 2) + 5,
        int(40 - (y_max - y_min) / 2) - 5:int(40 + (y_max - y_min) / 2) + 5] = ori_img[x_min - 5:x_max + 5,
                                                                               y_min - 5:y_max + 5]
        crop_img_r[int(40 - (x_max - x_min) / 2) - 5:int(40 + (x_max - x_min) / 2) + 5,
        int(40 - (y_max - y_min) / 2) - 5:int(40 + (y_max - y_min) / 2) + 5] = roi_img[x_min - 5:x_max + 5,
                                                                               y_min - 5:y_max + 5]
        # crop_img_r[int(40 - (x_max - x_min) / 2) - 5:int(40 + (x_max - x_min) / 2) + 5,
        # int(40 - (y_max - y_min) / 2) - 5:int(40 + (y_max - y_min) / 2) + 5] = 1
        return crop_img_t, crop_img_r

    # 提取感兴趣区域对应图像的特征，输出特征和对应样本名的列表；
    # save_roi表示是否要把roi图像保存下来，是为了看提取的特征区域是否正确；
    # name_loc是指定样本名在路径中的位置，-3是指倒数第三个文件夹名是样本名，例如：./DATA/T1M\an_qun_ying_054Y\1.dcm
    def get_feature_table_1(self, dcm_file_in, roi_file_in, save_roi=False, jpg_save_path='roi_img', name_loc=-2):
        result_all = []
        name_all = []
        nii_miss = []
        img_save_array = []
        for i in range(len(dcm_file_in)):
            # if i < 10:
            print(dcm_file_in[i])
            name_p = dcm_file_in[i].split("\\")[name_loc]
            img_t = sitk.GetArrayFromImage(sitk.ReadImage(dcm_file_in[i]))[0, :, :]
            img_r = sitk.GetArrayFromImage(sitk.ReadImage(roi_file_in[i]))[-1, :, :]
            crop_img_t, crop_img_r = self.crop_img(img_t, img_r)
            # plt.imshow(img_r, cmap='gray')
            # plt.show()
            if save_roi:
                self.save_roi_img(crop_img_t, crop_img_r, name_p, jpg_save_path)
                if np.sum(crop_img_r) > 0:
                    name_all.append(name_p)
                    img = sitk.GetImageFromArray(crop_img_t)
                    mask = sitk.GetImageFromArray(crop_img_r)
                    result = self.extractor.execute(img, mask)
                    img_save_array.append(crop_img_t*crop_img_r)
                    # img_save_array.append(crop_img_t)
                    # plt.imshow(np.concatenate((crop_img_t, crop_img_t*crop_img_r), axis=1), cmap='gray')
                    # plt.show()
                    result_all.append(result)
                else:
                    nii_miss.append(dcm_file_in[i])
        return result_all, name_all, nii_miss, img_save_array

    def feature_extract(self, level):
        files = self.get_file(self.path_img)  # 得到文件夹下所有的子文件路径的列表

        dcm_file_in, roi_file_in, miss_patient_1 = self.devide_img_roi(files, level+'.nii')  # 分开原始图片和感兴趣区域文件，这里只放进后缀为“1.nii”和“1.dcm”的文件
        result_all, name_all, nii_miss_1, img_save_array_1 = self.get_feature_table_1(dcm_file_in, roi_file_in,
                                                                                 save_roi=True,
                                                                                 jpg_save_path=self.jpg_save_path,
                                                                                 name_loc=-2)  # 提取感兴趣区域对应图像的特征

        np.save(self.img_save_path, img_save_array_1)
        np.save(self.name_path, name_all)
        df = self.result_to_table(result_all, name_all)  # 把得到的特征存入一个dataframe中
        df.to_csv(self.radiomic_result_path)
        return nii_miss_1


def make_dir(path):
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


if __name__ == '__main__':
    path_image = './Normal control_T1M'
    path_param = './MR_2D_extraction.yaml'  # params文件路径

    # HCM 心尖
    path_save_jpg = './img_jpg/control_jpg_img_1'
    path_save_img = './roi_npy/control_roi_img_1.npy'
    path_radiomic_result = './results/without_shape/control_df_label_1.csv'
    path_name = './roi_npy/control_name_img_1.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, param_path=path_param, jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, name_path=path_name)

    nii_miss = preprocess.feature_extract(level='1')
    print('nii_miss_1', nii_miss)

    # HCM 中间
    path_save_jpg = './img_jpg/control_jpg_img_2'
    path_save_img = './roi_npy/control_roi_img_2.npy'
    path_radiomic_result = './results/without_shape/control_df_label_2.csv'
    path_name = './roi_npy/control_name_img_2.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, param_path=path_param, jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='2')
    print('nii_miss_2', nii_miss)

    # HCM 心底
    label_name = '心底'
    path_save_jpg = './img_jpg/control_jpg_img_3'
    path_save_img = './roi_npy/control_roi_img_3.npy'
    path_radiomic_result = './results/without_shape/control_df_label_3.csv'
    path_name = './roi_npy/control_name_img_3.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, param_path=path_param,
                           jpg_save_path=path_save_jpg, img_save_path=path_save_img,radiomic_result_path=path_radiomic_result, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='3')
    print('nii_miss_3', nii_miss)








