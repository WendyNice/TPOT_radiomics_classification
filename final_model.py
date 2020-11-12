import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns

from sklearn.model_selection import KFold
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier


# Measurement of TP, FP, TN, FN
def perf_measure(true_label, predict_label):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(predict_label)):
        if true_label[i] == predict_label[i] == 1:
           TP += 1
        if predict_label[i] == 1 and true_label[i] == 0:
           FP += 1
        if true_label[i] == predict_label[i] == 0:
           TN += 1
        if predict_label[i] == 0 and true_label[i] == 1:
           FN += 1
    return TP, FP, TN, FN


# Evaluation metrics measurements.
def evaluation(TP, FP, TN, FN):
    # Sensitivity, hit rate, recall, or true positive rate
    Sensitivity = round(TP / (TP + FN), 4)
    # Specificity or true negative rate
    Specificity = round(TN / (TN + FP), 4)
    # Overall accuracy
    Accuracy = round((TP + TN) / (TP + FP + FN + TN), 4)
    # Precision or positive predictive value
    Precision = round(TP / (TP + FP), 4)
    # F1-score
    F1_score = round((2*Precision*Sensitivity)/(Precision + Sensitivity), 4)
    return Sensitivity, Specificity, Accuracy, Precision, F1_score


def plot_confusion_matrix(true_label, predict_label, dir_save):
    confusion = confusion_matrix(true_label, predict_label)
    print('confusion', confusion)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # plt.xticks(indices, [0, 1, 2])
    # plt.yticks(indices, [0, 1, 2])
    plt.xticks(indices, ['0', '1'])
    plt.yticks(indices, ['0', '1'])
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('confusion matrix')
    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 显示数据
    plt.text(0, 0, confusion[0][0])
    plt.text(0, 1, confusion[1][0])
    plt.text(1, 0, confusion[0][1])
    plt.text(1, 1, confusion[1][1])
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.savefig(os.path.join(dir_save, 'confusion_matrix.png'), dpi=300)
    plt.close()


def autolabel(rects, Num=1.12, rotation1=90, NN=1):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() - 0.04 + rect.get_width() / 2., Num * height, '%s' % float(height * NN),
                 rotation=rotation1)


def plot_bar(categories, values, dir_save):
    rects1 = plt.bar(categories, values)  # 横放条形图函数 barh
    plt.title('Classification indexes')
    plt.ylim(0, 1)
    autolabel(rects1, 1.09)
    plt.savefig(os.path.join(dir_save, 'classification_indexes.png'), dpi=300)
    plt.close()


def plot_roc(training_target, results_train_probe, testing_target, results_probe, dir_save):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(training_target, results_train_probe)
    roc_auc_train = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    # plt.plot(false_positive_rate, true_positive_rate, 'b--', label='Train AUC = %0.4f' % roc_auc_train)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(testing_target, results_probe)
    roc_auc_test = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, 'g--', label='Test AUC = %0.4f' % roc_auc_test)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(os.path.join(dir_save, 'roc.png'), dpi=300)
    plt.close()
    return roc_auc_train, roc_auc_test


def heatmap(df, xlabel, ylabel, dir_save):
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    # feature_num = [i + 1 for i in range(10)]
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(np.array(df), xticklabels=xlabel, yticklabels=ylabel, cmap=cmap)
    plt.savefig(os.path.join(dir_save, 'heapmap.png'), dpi=300)
    plt.close()

from sklearn.ensemble import ExtraTreesClassifier

# # LGE+ VS LGE-
# exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=3, max_features=0.45, min_samples_leaf=2,
#                                                min_samples_split=13, n_estimators=100, subsample=0.7000000000000001, random_state=33)


# # con vs lge-


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.1, min_samples_leaf=1, min_samples_split=8, n_estimators=100)

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=8, min_child_weight=17, n_estimators=100, nthread=1, subsample=0.4)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.001, max_depth=1, min_child_weight=13, n_estimators=100, nthread=1, subsample=0.35000000000000003)),
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=1, min_samples_split=8, n_estimators=100, random_state=33)
)
# # con vs HCM
# exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=3, max_features=0.45, min_samples_leaf=14,
#                                                min_samples_split=6, n_estimators=100, subsample=0.7000000000000001, random_state=23)

data_path = './data_norm/con_vs_lge-'
feature_train = np.load(os.path.join(data_path, 'feature_train.npy'))
label_train = np.load(os.path.join(data_path, 'label_train.npy'))
feature_test = np.load(os.path.join(data_path, 'feature_test.npy'))
label_test = np.load(os.path.join(data_path, 'label_test.npy'))

print(label_train.shape, label_test.shape)
count_num = Counter(np.array(label_train))
print(count_num)
count_num = Counter(np.array(label_test))
print(count_num)

exported_pipeline.fit(feature_train, label_train)
score = exported_pipeline[-1].feature_importances_
# score = exported_pipeline[-1].feature_importances_

print('score', score)
loc = list(np.where(score > 0)[0])
print('loc', len(loc), loc)
loc_val = list(score[loc])
# loc_val rank num
sort_index = sorted(range(len(loc_val)), key=lambda k: loc_val[k])
print('sort_index', len(sort_index), sort_index)

# print(loc_val)
top_feature_index_15 = sort_index[-15:]
top_feature_index_10 = sort_index[-5:]

print('top_feature_index_10', top_feature_index_10)

print('loc_val', loc_val)
value_10 = [round(loc_val[i], 4) for i in top_feature_index_10]
print('feature_value', value_10)
value_10.sort(reverse=True)
value_15 = [loc_val[i] for i in top_feature_index_15]
value_15.sort(reverse=True)

print(value_15)

# index of loc
final_index = [loc[i] for i in top_feature_index_10]
# print('final_index', final_index)
feature_name = np.load(os.path.join(data_path, 'feature_name.npy'), allow_pickle=True)
feature_select = [feature_name[i] for i in top_feature_index_10]
print('feature_name', feature_select)

df = feature_test[:, final_index]
df_norm = []

for i in range(5):
    data = df[:, i]
    # print(data.shape)
    list_norm = (data - np.min(data))/(np.max(data) - np.min(data))
    df_norm.append(list_norm)
df_norm = np.array(df_norm)
print(df_norm.shape)

dir_save = os.path.join('./plots', 'con_vs_lge-')
# dir_save = os.path.join('./plots', 'con_vs_lge-')
# dir_save = os.path.join('./plots', 'con_vs_lge-_19_6')
if not os.path.exists(dir_save):
    os.makedirs(dir_save)

# hot plot
feature_num = [i + 1 for i in range(5)]
np.save('./heatmap/con_vs_lge-/df.npy', df_norm)
np.save('./heatmap/con_vs_lge-/label.npy', label_test)
np.save('./heatmap/con_vs_lge-/feature_select.npy', feature_select)
print('feature_select', feature_select)
heatmap(df_norm, xlabel=label_test, ylabel=feature_num, dir_save=dir_save)

print('Done ')


train_result = exported_pipeline.predict(feature_train)
train_result_probe = exported_pipeline.predict_proba(feature_train)[:, 1]

test_result = exported_pipeline.predict(feature_test)
test_result_probe = exported_pipeline.predict_proba(feature_test)[:, 1]

print('##########-test_result-##################')

TP, FP, TN, FN = perf_measure(label_test, test_result)
Sensitivity, Specificity, Accuracy, Precision, F1_score = evaluation(TP, FP, TN, FN)
bar_categories = ['Sen', 'Spe', 'Acc', 'Pre', 'F1_s']
bar_values = Sensitivity, Specificity, Accuracy, Precision, F1_score
plot_confusion_matrix(label_test, test_result, dir_save)
plot_bar(bar_categories, bar_values, dir_save)
print('Accuracy', Accuracy)
print('Sensitivity', Sensitivity)
print('Specificity', Specificity)
print('Precision', Precision)


# print(label_train.shape, train_result_probe.shape, label_test.shape, test_result_probe.shape)
roc_auc_train, roc_auc_test = plot_roc(label_train, train_result_probe, label_test, test_result_probe,
                                       dir_save)
print('AUC', roc_auc_test)



