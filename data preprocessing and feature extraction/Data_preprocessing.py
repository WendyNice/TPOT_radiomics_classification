# 做机器学习分类的一些小感悟：
# 一开始就定好random_seed，分训练集/测试机/验证机，且验证集要到最后确定特征和模型后才能进行验证。
# 看训练集和测试集的结果调参。
# 如果样本的数据量太少，特征的维度较大时容易过拟合，这时可以进行特征降维
# 只要测试集的样本基本正负均衡，测试集的结果很大程度上已经可以判断模型的效果了，不需要改变随机数进行训练集和测试机的重新分配
# 模型的结果很大程度上还是由样本量决定的，如果结果没有过拟合，太多的调参对效果的提高不大

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2
from sklearn import preprocessing
from feature_selector import FeatureSelector
from sklearn.ensemble import RandomForestClassifier
import os

# 特征筛选
def select_feature(feature, label):
    fs = FeatureSelector(data=feature, labels=label)
    # Missing value
    fs.identify_missing(missing_threshold=0.3)
    # Single_unique_value
    fs.identify_single_unique()
    # Collinear value
    # fs.identify_collinear(correlation_threshold=0.98, one_hot=False)
    # fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=20)
    # fs.identify_low_importance(cumulative_importance=0.99)
    invalid_feature_d = fs.ops  # 字典
    invalid_feature = []
    for k, v in invalid_feature_d.items():
        for feature in v:
            if feature not in invalid_feature:
                invalid_feature.append(feature)
    return invalid_feature


# 标准化
def standardscaler_feature(train_feature):
    columns_name = train_feature.columns
    scaler = preprocessing.StandardScaler().fit(train_feature)
    return columns_name, scaler


# method: f_clssif/chi2/RFE
# f_clssif: Compute the ANOVA F-value for the provided sample.
def select_feature_sklearn(X_train, label, feature_name, method, feature_num):
    if method == 'f_classif' or method == 'chi2':
        if method == 'f_classif':
            fs = SelectKBest(f_classif, k=feature_num)
        else:
            fs = SelectKBest(chi2, k=feature_num)

        fs.fit(X_train, label)
        feature_n = fs.get_support(True)
        # print('feature_n', feature_n)
        feature_select = [feature_name[i] for i in range(len(feature_name)) if i in feature_n]
        print('feature_select', feature_select)
    else:
        fs = RFE(DecisionTreeClassifier(), n_features_to_select=feature_num)
        fs.fit(X_train, label)
        feature_n = fs.support_
        feature_select = np.where(feature_n == True)
        feature = feature_select[0].tolist()
        print('feature', feature)
        feature_se = []
        for j in range(len((feature_name).tolist())):
            if j in feature:
                feature_se.append(((feature_name).tolist())[j])
        print('feature_select', feature_se)
    return feature_n


# 分类
# Classifier: DecisionTreeClassifier/SVC/RandomForestClassifier/LogisticRegression
def classify_analysis(X_train, y_train, method='SVC'):
    clf = []
    print('method', method)
    if method == 'DecisionTreeClassifier':
        clf = DecisionTreeClassifier()
    if method == 'SVC':
        clf = SVC(kernel="linear", probability=True)
    if method == 'RandomForestClassifier':
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    if method == 'LogisticRegression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def clf_result(clf, feature, label):
    print('accuracy_score')
    print(accuracy_score(label, clf.predict(feature)))
    print(classification_report(label, clf.predict(feature)))


# 画散点图
def plot_dot(features_1, features_2, label):
    plt.figure()  # 新建一张图进行绘制
    plt.scatter(features_1, features_2, c=label, edgecolor='k')  # 绘制两个主成分组成坐标的散点图
    plt.show()


# 主成分分析
def pca_anlysis(feature_train, feature_test, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(feature_train)
    features_pca_train = pca.fit_transform(feature_train)
    features_pca_test = pca.fit_transform(feature_test)
    return features_pca_train, features_pca_test


# select_sklearn: whether to conduct further feature screening
# method_sk: the method of further feature screening (f_clssif/chi2/RFE)
# method_classification: classification method (DecisionTreeClassifier/SVC/RandomForestClassifier/LogisticRegression)
# feature_number: the number of the selected features
def feature_select_and_predict(save_path, X_train, y_train, X_test, y_test, select_sklearn=False, method_sk='f_classif', feature_num=10,
                           method_classification='DecisionTreeClassifier'):
    # 特征筛选
    print('-------------Preliminary feature screening---------------')
    invalid_feature = select_feature(X_train, y_train)
    X_train = X_train.drop(invalid_feature, axis=1)
    X_test = X_test.drop(invalid_feature, axis=1)
    print('X_train.columns', X_train.columns)
    np.save(os.path.join(save_path, 'feature_name.npy'), X_train.columns)

    # 标准化
    columns_name, scaler = standardscaler_feature(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    np.save(os.path.join(save_path, 'feature_train.npy'), X_train)
    np.save(os.path.join(save_path, 'feature_test.npy'), X_test)

    # if select_sklearn == True:
    #     print('--------------Further feature screening---------------')
    #     feature_n = select_feature_sklearn(X_train, y_train,
    #                                                       columns_name, method=method_sk, feature_num=feature_num)
    #     X_train = X_train[:, feature_n]
    #     X_test = X_test[:, feature_n]
    #     print('X_train', X_train.shape)
    #     print('X_test', X_test.shape)
    # clf = classify_analysis(X_train, y_train, method=method_classification)
    # print("--------Train set result-----------")
    # clf_result(clf, X_train, y_train)
    # print("--------Test set result-----------")
    # clf_result(clf, X_test, y_test)

    # PCA
    # features_t, features_t_test = pca_anlysis(X_train, X_test, n_components=2)
    # print('PCA results：')
    # clf_1 = classify_analysis(features_t, y_train method=method_classification)
    # print("--------Train set result-----------")
    # clf_result(clf_1, features_t, y_train)
    # print("--------Test set result-----------")
    # clf_result(clf_1, features_t_test, y_test)
    # plot_dot(features_t[:, 0], features_t[:, 1], y_train)
    print('----------------end---------------------')


data_path = './data/lge+_vs_lge-'
feature_train = np.load(os.path.join(data_path, 'feature_train.npy'))
label_train = np.load(os.path.join(data_path, 'label_train.npy'))
feature_test = np.load(os.path.join(data_path, 'feature_test.npy'))
label_test = np.load(os.path.join(data_path, 'label_test.npy'))

print('feature_train', feature_train.shape)
print('feature_test', feature_test.shape)
table_ori = pd.read_csv(r'D:\Data analysis\HCM\9_4\results\without_shape\control_df_label_1.csv')
feature_name = table_ori.columns.tolist()[1:]
print(len(feature_name))
feature_train = pd.DataFrame(feature_train, columns=feature_name)
feature_test = pd.DataFrame(feature_test, columns=feature_name)
# Further feature screening: select_sklearn=True
# Further feature screening method: method_sk='f_classif',
# Classification method: method_classification='SVC'
# Selected feature number: feature_num=10
save_path = r'./data_norm/lge+_vs_lge-'
feature_select_and_predict(save_path, feature_train, label_train, feature_test, label_test, select_sklearn=True, method_sk='f_classif', feature_num=10,
                             method_classification='SVC')

