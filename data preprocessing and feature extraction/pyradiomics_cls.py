'''
Tpot为数据集找到最好的pipeline
By: Haixia

安装Tpot: pip install tpot
安装xgboost: conda install py-xgboost  # 将许多弱分类器集成在一起，形成一个强分类器
安装xgboost可能需要重装TPOT和scikit-learn
'''

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
import random
from sklearn.model_selection import KFold
import time
import os


def tpot_model_cv(X_train, y_train, save_path, cv_num):
    print('Cross validation')
    print(X_train.shape, y_train.shape)
    count = 0
    KF = KFold(n_splits=cv_num)  # 建立4折交叉验证方法 查一下KFold函数的参数
    for train_index, val_index in KF.split(X_train):
        count += 1
        print('count', count)
        x_train_sub, x_val = X_train[train_index], X_train[val_index]
        y_train_sub, y_val = y_train[train_index], y_train[val_index]
        print(x_train_sub.shape, y_train_sub.shape)
        print(x_val.shape, y_val.shape)
        count_num = Counter(np.array(y_train_sub))
        print(count_num)
        count_num = Counter(np.array(y_val))
        print(count_num)
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=33, n_jobs=16)
        tpot.fit(x_train_sub, y_train_sub)
        print(tpot.score(x_val, y_val))
        tpot.export(save_path + str(count) + '.py')


def tpot_model(X_train, y_train, save_path):
    print(X_train.shape, y_train.shape)
    count_num = Counter(np.array(y_train))
    print(count_num)
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=30, n_jobs=16)
    tpot.fit(X_train, y_train)
    tpot.export(save_path + '1.py')

# # 修改生成的TPOT文件
# def alter_content(file, file_alter, cv_num):
#     with open(file_alter, 'w', encoding="utf-8") as f_a:
#         f_a.write('results_file = \'./results.txt\'')
#         f_a.write('\n')
#         f_a.write('results_train_file = \'./results_train.txt\'')
#         f_a.write('\n')
#         f_a.write('with open(results_file, \'w\', encoding="utf-8") as f_f:')
#         f_a.write('\n')
#         f_a.write('    f_f.write(\'Pipelines results:\')')
#         f_a.write('\n')
#         f_a.write('    f_f.write(\'\\n\')')
#         f_a.write('\n')
#         f_a.write('with open(results_train_file, \'w\', encoding="utf-8") as f_f:')
#         f_a.write('\n')
#         f_a.write('    f_f.write(\'Pipelines results:\')')
#         f_a.write('\n')
#         f_a.write('    f_f.write(\'\\n\')')
#         f_a.write('\n')
#         f_a.write('\n')
#         f_a.write('\n')
#         for i in range(cv_num):
#             file_name = file + str(i+1) + '.py'
#             f_a.write('from sklearn.metrics import accuracy_score, classification_report')
#             f_a.write('\n')
#             with open(file_name, "r", encoding="utf-8") as f:
#                 for line in f:
#                     if 'tpot_data = ' in line:
#                         line_0 = 'training_features = np.load(\'../data/train_test_split/feature_train.npy\')'
#                         line_1 = 'training_target = np.load(\'../data/train_test_split/label_train.npy\')'
#                         line_2 = 'testing_features = np.load(\'../data/train_test_split/feature_test.npy\')'
#                         line_3 = 'testing_target = np.load(\'../data/train_test_split/label_test.npy\')'
#                         f_a.write(line_0)
#                         f_a.write('\n')
#                         f_a.write(line_1)
#                         f_a.write('\n')
#                         f_a.write(line_2)
#                         f_a.write('\n')
#                         f_a.write(line_3)
#                         f_a.write('\n')
#                     if 'PATH/TO/DATA/FILE' not in line:
#                         if 'tpot_data.drop' not in line:
#                             if 'train_test_split(features, tpot_data[\'target\']' not in line:
#                                 if 'testing_target = \\' not in line:
#                                     if 'LinearSVC' not in line:
#                                         f_a.write(line)
#                                     else:
#                                         line = line.replace('LinearSVC', 'SVC').split(',')[0]
#                                         if line.startswith('from'):
#                                             f_a.write(line)
#                                         else:
#                                             line += ', probability=True'
#                                             f_a.write(line)
#                                             f_a.write(')')
#                                             f_a.write('\n')
#                     if 'results = exported_pipeline.predict(testing_features)' in line:
#                         f_a.write('results_probe = exported_pipeline.predict_proba(testing_features)[:, 1]')
#                         f_a.write('\n')
#                         f_a.write('results_train_probe = exported_pipeline.predict_proba(training_features)[:, 1]')
#                         f_a.write('\n')
#                 # f_a.write('print(accuracy_score(testing_target, results))')
#                 # f_a.write('\n')
#                 # f_a.write('print(classification_report(testing_target, results))')
#                 # f_a.write('\n')
#                 f_a.write('results_probe = list(results_probe)')
#                 f_a.write('\n')
#                 f_a.write('results_train_probe = list(results_train_probe)')
#                 f_a.write('\n')
#                 f_a.write('with open(results_file, \'a\', encoding="utf-8") as f_r:')
#                 f_a.write('\n')
#                 f_a.write('    f_r.write(str(results_probe))')
#                 f_a.write('\n')
#                 f_a.write('    f_r.write(\'\\n\')')
#                 f_a.write('\n')
#                 f_a.write('with open(results_train_file, \'a\', encoding="utf-8") as f_t:')
#                 f_a.write('\n')
#                 f_a.write('    f_t.write(str(results_train_probe))')
#                 f_a.write('\n')
#                 f_a.write('    f_t.write(\'\\n\')')
#                 f_a.write('\n')
#                 f_a.write('\n')
#                 f_a.write('\n')


# 修改生成的TPOT文件
def alter_content(file, file_alter, cv_num):
        for i in range(cv_num):
            # if i == 0:
                with open(file_alter + '_' + str(i+1) + '.py', 'w', encoding="utf-8") as f_a:
                    file_name = file + str(i+1) + '.py'
                    f_a.write('from sklearn.metrics import accuracy_score, classification_report')
                    f_a.write('\n')
                    with open(file_name, "r", encoding="utf-8") as f:
                        for line in f:
                            if 'PATH/TO/DATA/FILE' not in line:
                                if 'tpot_data.drop' not in line:
                                    if 'train_test_split(features, tpot_data[\'target\']' not in line:
                                        if 'testing_target = \\' not in line:
                                            if 'exported_pipeline.fit(training_features, training_target)' not in line:
                                                if 'results = exported_pipeline.' not in line:
                                                    f_a.write(line)
                                                    f_a.write('\n')
                merge_plot(file_alter + '_' + str(i+1) + '.py', file_plot, tpot_plot + '_' + str(i+1) + '.py')
                print('python ' + tpot_plot + '_' + str(i+1) + '.py' + ' --model_num=' + str(i + 1))
                py_plot = ('python ' + tpot_plot + '_' + str(i+1) + '.py' + ' --model_num=' + str(i + 1))
                os.system(py_plot)


def merge_plot(file_1, file_2, file_out):
    # print('file_1', file_1)
    # print('file_2', file_2)
    # print(os.getcwd())
    with open(file_out, 'w', encoding="utf-8") as f_m:
        with open(file_1, 'r', encoding="utf-8") as f_1:
            for line in f_1:
                # print('line', line)
                f_m.write(line)
            f_m.write('\n')
            f_m.write('\n')
        with open(file_2, 'r', encoding="utf-8") as f_2:
            for line in f_2:
                f_m.write(line)


def normalization(df):
    df_out = np.zeros((df.shape), dtype=float)
    for i in range(df.shape[1]):
        data = df[:, i]
        # print(data.shape)
        list_norm = (data - np.min(data))/(np.max(data) - np.min(data))
        df_out[:, i] = list_norm
    # print('df_out.shape', df_out.shape)
    # print('df_out', df_out)
    return df_out


time_start = time.time()
# prepare_path = './tpot_gen/prepare_data_1.py'
# file_path = './results/df_pyradiomics.csv'
save_path = './out_tpot/out_tpot_'
file_alter = './tpot_gen/Tpot_alter'
file_plot = './tpot_gen/cv_model.py'
tpot_plot = './tpot_gen/tpot_plot'
# file_result = './tpot_gen/ensemble_result.py'
cv_num = 10

# py_prepare = ('python ' + prepare_path)
# os.system(py_prepare)

data_path = './data_norm/lge+_vs_lge-'
feature_train = np.load(os.path.join(data_path, 'feature_train.npy'))
label_train = np.load(os.path.join(data_path, 'label_train.npy'))
feature_test = np.load(os.path.join(data_path, 'feature_test.npy'))
label_test = np.load(os.path.join(data_path, 'label_test.npy'))
pd.DataFrame(feature_train).to_csv('feature_train.csv')

feature_train = normalization(feature_train)
feature_test = normalization(feature_test)


if cv_num == 1:
    tpot_model(feature_train, label_train, save_path)  # X_train, X_test, y_train, y_test
else:
    tpot_model_cv(feature_train, label_train, save_path, cv_num)
time_here = time.time()
print('totally cost', time_here - time_start)
result = os.path.join('./plots', 'result.txt')
with open(result, 'w', encoding="utf-8") as f_r:
    f_r.write('/n')

alter_content(save_path, file_alter, cv_num)




#
#
# os.chdir('./tpot_gen')
# py_tpot = ('python ' + file_alter.split('/')[-1])
# os.system(py_tpot)
# merge_plot(file_result.split('/')[-1], file_plot.split('/')[-1], tpot_plot.split('/')[-1])
# py_plot = ('python ' + tpot_plot.split('/')[-1] + ' --cv_num=' + str(cv_num))
# os.system(py_plot)





