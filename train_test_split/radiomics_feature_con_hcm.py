from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import pandas as pd
from collections import Counter
import random
# 导入数据，train test split并整理成CNN可以导入的格式
# label list, 重复132个0，109个1wom
# control组标记为0，LGE阴性的组标记为1

dir = './results/without_shape/'

img1 = pd.read_csv(dir + 'control_df_label_1.csv')
img2 = pd.read_csv(dir + 'control_df_label_2.csv')
img3 = pd.read_csv(dir + 'control_df_label_3.csv')

img_1 = pd.concat([img1, img2, img3], axis=0)
label1 = [0 for i in range(img_1.shape[0])]
img4 = pd.read_csv(dir + 'no_LGE_df_label_1.csv')
img5 = pd.read_csv(dir + 'no_LGE_df_label_2.csv')
img6 = pd.read_csv(dir + 'no_LGE_df_label_3.csv')

img7 = pd.read_csv(dir + 'LGE_df_label_1.csv')
img8 = pd.read_csv(dir + 'LGE_df_label_2.csv')
img9 = pd.read_csv(dir + 'LGE_df_label_3.csv')
# img_2 = pd.concat([img4, img5, img6], axis=0)
img_2 = pd.concat([img4, img5, img6, img7, img8, img9], axis=0)
label2 = [1 for i in range(img_2.shape[0])]

print(img_1.shape, img_2.shape)
img = pd.concat([img_2, img_1], axis=0)
img = img.reset_index(drop=True)
label = label2 + label1
name = [i.replace('_', '').lower() for i in img['Unnamed: 0']]
# print('name', len(name), name)

name_train = np.load('./by_case/control_HCM_file_train.npy')
name_test = np.load('./by_case/control_HCM_file_test.npy')
name_val = np.load('./by_case/control_HCM_file_val.npy')
train_name_index = []
test_name_index = []
label_train = []
label_test = []

# print('name_train', name_train)
for i in range(len(name)):
    if name[i] in name_train:
        train_name_index.append(i)
        label_train.append(label[i])
    else:
        test_name_index.append(i)
        label_test.append(label[i])
print('train_name_index', len(train_name_index), train_name_index)
print('test_name_index', len(test_name_index), test_name_index)
print('label_train', len(label_train), label_train)
print('label_test', len(label_test), label_test)

label_train_0_index = [train_name_index[i] for i in range(len(train_name_index)) if label_train[i] == 0]
print('label_train_0_index', len(label_train_0_index))
label_train_1_index = [train_name_index[i] for i in range(len(train_name_index)) if label_train[i] == 1]
print('label_train_1_index', len(label_train_1_index))
random.seed(33)
label_train_1_index = random.sample(label_train_1_index, int(len(label_train_1_index)*0.5))
print('label_train_1_index', len(label_train_1_index), label_train_1_index)
label_train_0 = [0 for i in range(len(label_train_0_index))]
label_train_1 = [1 for i in range(len(label_train_1_index))]
train_name_index = label_train_0_index + label_train_1_index
label_train = label_train_0 + label_train_1

train_df = img.iloc[train_name_index]
train_df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
name_train_df = train_df['name']
train_df = train_df[[i for i in train_df.columns.tolist()[1:]]]

# list_select_1 = [i for i in train_df.columns.tolist() if 'glrlm' in i or 'firstorder' in i]
# train_df = train_df[list_select_1]

test_df = img.iloc[test_name_index]
test_df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
name_test_df = test_df['name']
test_df = test_df[[i for i in test_df.columns.tolist()[1:]]]
# test_df = test_df[list_select_1]
# print(train_df)
# print(test_df)

train_df = np.array(train_df)
label_train = np.array(label_train)
random.seed(33)

index = [i for i in range(label_train.shape[0])]
random.shuffle(index)  # 打乱索引
train_df = train_df[index]
label_train = label_train[index]
print('label_train.shape[0]', label_train.shape[0])
print('label_test.shape[0]', np.array(label_test).shape[0])

np.save(r'D:\Data analysis\Tpot plugin\demo\data\train_test_split\feature_train.npy', train_df)
np.save(r'D:\Data analysis\Tpot plugin\demo\data\train_test_split\feature_test.npy', np.array(test_df))
np.save(r'D:\Data analysis\Tpot plugin\demo\data\train_test_split\label_train.npy', label_train)
np.save(r'D:\Data analysis\Tpot plugin\demo\data\train_test_split\label_test.npy', np.array(label_test))

# # validation
# label_train_npy = np.load('./by_case/train_label_npy.npy')
# print(list(label_train_npy) == label_train)
# print(list(label_train_npy))
# print(label_train)
# label_test_npy = np.load('./by_case/test_label_npy.npy')
# print(list(label_test_npy) == label_test)



