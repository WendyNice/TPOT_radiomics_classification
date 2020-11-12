'''
Label: Huaxi HCM 3 level
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
    def __init__(self, path_img, label_path, param_path, jpg_save_path, img_save_path, radiomic_result_path, label_col, name_path):
        self.path_img = path_img
        self.label_path = label_path
        self.param_path = param_path
        self.jpg_save_path = jpg_save_path
        self.img_save_path = img_save_path
        self.radiomic_result_path = radiomic_result_path
        self.label_col = label_col
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
    def devide_img_roi(files, name_sel, num):
        roi_file = [i for i in files if i.endswith(num)]
        roi_file_sel = []
        for name in roi_file:
            if name.split("\\")[-2].replace("_", "").replace(" ", "").lower() in name_sel:
                roi_file_sel.append(name)
        dcm_file = [i.replace(".nii", ".dcm") for i in roi_file_sel]
        dcm_file_in = []
        roi_file_in = []

        for f in range(len(dcm_file)):
            if os.path.exists(dcm_file[f]):
                dcm_file_in.append(dcm_file[f])
                roi_file_in.append(roi_file_sel[f])
        print('len(dcm_file_in), len(roi_file_in)', len(dcm_file_in), len(roi_file_in))
        return dcm_file_in, roi_file_in

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
        crop_img_t[int(40-(x_max-x_min)/2)-5:int(40+(x_max-x_min)/2)+5, int(40-(y_max-y_min)/2)-5:int(40+(y_max-y_min)/2)+5] = ori_img[x_min-5:x_max+5, y_min-5:y_max+5]
        crop_img_r[int(40-(x_max-x_min)/2)-5:int(40+(x_max-x_min)/2)+5, int(40-(y_max-y_min)/2)-5:int(40+(y_max-y_min)/2)+5] = roi_img[x_min-5:x_max+5, y_min-5:y_max+5]
        # crop_img_r[int(40-(x_max-x_min)/2)-5:int(40+(x_max-x_min)/2)+5, int(40-(y_max-y_min)/2)-5:int(40+(y_max-y_min)/2)+5] = 1
        # plt.imshow(crop_img_r)
        # plt.show()
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
            print(dcm_file_in[i])
            name_p = dcm_file_in[i].split("\\")[name_loc]
            print('name', name_p)
            img_t = sitk.GetArrayFromImage(sitk.ReadImage(dcm_file_in[i]))[0, :, :]
            img_r = sitk.GetArrayFromImage(sitk.ReadImage(roi_file_in[i]))[-1, :, :]
            crop_img_t, crop_img_r = self.crop_img(img_t, img_r)
            # plt.imshow(crop_img_t, cmap='gray')
            # plt.show()
            if save_roi:
                self.save_roi_img(crop_img_t, crop_img_r, name_p, jpg_save_path)
            if np.sum(crop_img_r) > 0:
                name_all.append(name_p)
                img = sitk.GetImageFromArray(crop_img_t)
                mask = sitk.GetImageFromArray(crop_img_r )
                result = self.extractor.execute(img, mask)
                img_save_array.append(crop_img_t*crop_img_r)
                # img_save_array.append(crop_img_t)
                # plt.imshow(np.concatenate((crop_img_t, crop_img_t*crop_img_r), axis=1), cmap='gray')
                # plt.show()
                result_all.append(result)
            else:
                nii_miss.append(dcm_file_in[i])
        return result_all, name_all, nii_miss, img_save_array

    def find_label(self, label=1, name_col='姓名'):
        name_sel = []
        label_df = pd.read_excel(self.label_path)
        label_df.columns = [str(i) for i in label_df.columns]
        name = [i.replace(" ", "").replace("_", "").lower() for i in label_df[name_col].tolist()]
        label_sel = label_df[self.label_col].tolist()
        # print('label_df[str(i)]', label_df[str(i)])
        for j in range(len(label_sel)):
            if label_sel[j] == label:
                name_sel.append(name[j])
        print('name_sel', name_sel)
        return name_sel

    def feature_extract(self, level, LGE=1):
        files = self.get_file(self.path_img)  # 得到文件夹下所有的子文件路径的列表
        name_sel = self.find_label(label=LGE, name_col='姓名')
        dcm_file_in, roi_file_in = self.devide_img_roi(files, name_sel, level+'.nii')  # 分开原始图片和感兴趣区域文件，这里只放进后缀为“1.nii”和“1.dcm”的文件
        result_all, name_all, nii_miss_1, img_save_array_1 = self.get_feature_table_1(dcm_file_in, roi_file_in,
                                                                                 save_roi=True,
                                                                                 jpg_save_path=self.jpg_save_path,
                                                                                 name_loc=-2)  # 提取感兴趣区域对应图像的特征
        df = self.result_to_table(result_all, name_all)  # 把得到的特征存入一个dataframe中
        df.to_csv(self.radiomic_result_path)
        np.save(self.img_save_path, img_save_array_1)
        np.save(self.name_path, name_all)
        return nii_miss_1


def make_dir(path):
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


if __name__ == '__main__':
    path_image = './T1M 200903'
    path_label = './Label.xlsx'
    path_param = './MR_2D_extraction.yaml'  # params文件路径
    make_dir('./img_jpg')
    make_dir('./roi_npy')
    make_dir('./results/without_shape')

    ####################################################
    # HCM LGE 心尖
    label_name = '心尖'

    path_save_jpg = './img_jpg/LGE_jpg_img_1'
    path_save_img = './roi_npy/LGE_roi_img_1.npy'
    path_radiomic_result = './results/without_shape/LGE_df_label_1.csv'
    path_name = './roi_npy/LGE_name_img_1.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, label_path=path_label, param_path=path_param, jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, label_col=label_name, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='1', LGE=1)
    print('nii_miss_1', nii_miss)

    # HCM no_LGE 心尖
    path_save_jpg = './img_jpg/no_LGE_jpg_img_1'
    path_save_img = './roi_npy/no_LGE_roi_img_1.npy'
    path_radiomic_result = './results/without_shape/no_LGE_df_label_1.csv'
    path_name = './roi_npy/no_LGE_name_img_1.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, label_path=path_label, param_path=path_param,
                           jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, label_col=label_name, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='1', LGE=0)
    print('nii_miss_1', nii_miss)

    ####################################################
    # HCM LGE 中间
    label_name = '中间'

    path_save_jpg = './img_jpg/LGE_jpg_img_2'
    path_save_img = './roi_npy/LGE_roi_img_2.npy'
    path_radiomic_result = './results/without_shape/LGE_df_label_2.csv'
    path_name = './roi_npy/LGE_name_img_2.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, label_path=path_label, param_path=path_param,
                           jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, label_col=label_name, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='2', LGE=1)
    print('nii_miss_2', nii_miss)

    # HCM no_LGE 中间
    path_save_jpg = './img_jpg/no_LGE_jpg_img_2'
    path_save_img = './roi_npy/no_LGE_roi_img_2.npy'
    path_radiomic_result = './results/without_shape/no_LGE_df_label_2.csv'
    path_name = './roi_npy/no_LGE_name_img_2.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, label_path=path_label, param_path=path_param,
                           jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, label_col=label_name, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='2', LGE=0)
    print('nii_miss_2', nii_miss)

    # HCM LGE 心底
    label_name = '心底'
    path_save_jpg = './img_jpg/LGE_jpg_img_3'
    path_save_img = './roi_npy/LGE_roi_img_3.npy'
    path_radiomic_result = './results/without_shape/LGE_df_label_3.csv'
    path_name = './roi_npy/LGE_name_img_3.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, label_path=path_label, param_path=path_param,
                           jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, label_col=label_name, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='3', LGE=1)
    print('nii_miss_3', nii_miss)

    # HCM no_LGE 心底
    label_name = '心底'
    path_save_jpg = './img_jpg/no_LGE_jpg_img_3'
    path_save_img = './roi_npy/no_LGE_roi_img_3.npy'
    path_radiomic_result = './results/no_LGE_df_label_3.csv'
    path_name = './roi_npy/no_LGE_name_img_3.npy'

    make_dir(path_save_jpg)

    preprocess = Preproces(path_img=path_image, label_path=path_label, param_path=path_param,
                           jpg_save_path=path_save_jpg, img_save_path=path_save_img,
                           radiomic_result_path=path_radiomic_result, label_col=label_name, name_path=path_name)
    nii_miss = preprocess.feature_extract(level='3', LGE=0)
    print('nii_miss_3', nii_miss)








