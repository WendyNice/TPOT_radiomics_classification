import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


# con vs lge-
def heatmap(df, xlabel, ylabel):
    cmap = 'Blues'
    feature_num = [i + 1 for i in range(10)]
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(np.array(df), xticklabels=xlabel, yticklabels=ylabel, cmap=cmap)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 35,
             }
    ax.set_xlabel('Labels', font2)
    ax.set_ylabel('Features', font2)
    # plt.title('con vs lge-', fontsize=40, fontweight='bold', loc ='left', verticalalignment='button')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=25)
    plt.subplots_adjust(left=0.5, bottom=0.2, right=1, top=0.9, hspace=0.1, wspace=0.1)
    plt.axhline(y=5, xmin=0, xmax=0.476, c="r", ls="-", lw=10)
    plt.axhline(y=5, xmin=0.489, xmax=1, c="g", ls="-", lw=10)
    plt.legend(['LGE-', 'Control'], fontsize=20)
    plt.savefig('./heatmap/con_vs_lge-.jpg', dpi=300)
    plt.show()
#

df_norm = np.load('./heatmap/con_vs_lge-/df.npy')
label_test = np.load('./heatmap/con_vs_lge-/label.npy')
feature_select = np.load('./heatmap/con_vs_lge-/feature_select.npy')
# heatmap(df_norm, xlabel=label_test, ylabel=feature_select)
print('feature_select', feature_select)

# con_vs_hcm
def heatmap(df, xlabel, ylabel):
    cmap = 'Blues'
    feature_num = [i + 1 for i in range(10)]
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(np.array(df), xticklabels=xlabel, yticklabels=ylabel, cmap=cmap)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 35,
             }
    ax.set_xlabel('Labels', font2)
    ax.set_ylabel('Features', font2)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=25)
    plt.subplots_adjust(left=0.4, bottom=0.2, right=1, top=0.9, hspace=0.1, wspace=0.1)
    plt.axhline(y=5, xmin=0, xmax=1, c="r", ls="-", lw=10)
    plt.axhline(y=5, xmin=0.69, xmax=1, c="g", ls="-", lw=10)
    plt.legend(['HCM', 'Control'], fontsize=20)
    plt.savefig('./heatmap/con_vs_hcm.jpg', dpi=300)
    plt.show()


# df_norm = np.load('./heatmap/con_vs_hcm/df.npy')
# label_test = np.load('./heatmap/con_vs_hcm/label.npy')
feature_select = np.load('./heatmap/con_vs_hcm/feature_select.npy')
# heatmap(df_norm, xlabel=label_test, ylabel=feature_select)
#
print('feature_select', feature_select)


from collections import Counter
# LGE+ vs lge-
def heatmap(df, xlabel, ylabel):
    cmap = 'Blues'
    feature_num = [i + 1 for i in range(10)]
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(np.array(df), xticklabels=xlabel, yticklabels=ylabel, cmap=cmap)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 35,
             }
    ax.set_xlabel('Labels', font2)
    ax.set_ylabel('Features', font2)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=25)
    plt.subplots_adjust(left=0.5, bottom=0.2, right=1, top=0.9, hspace=0.1, wspace=0.1)
    plt.axhline(y=5, xmin=0, xmax=1, c="r", ls="-", lw=10)
    plt.axhline(y=5, xmin=0.557, xmax=1, c="g", ls="-", lw=10)
    plt.legend(['LGE-', 'LGE+'], fontsize=20)
    plt.savefig('./heatmap/lge+_vs_lge-.jpg', dpi=300)
    plt.show()

# import pandas as pd
# df_norm = pd.DataFrame(np.load('./heatmap/lge+_vs_lge-/df.npy'))
# print(df_norm)
#
# label_test = np.load('./heatmap/lge+_vs_lge-/label.npy')
# print(label_test)
#
# count_num = Counter(np.array(label_test))
# print(count_num)
# order_0 = [i for i in range(38)]
# order_1 = [i for i in range(38, len(label_test))]
#
# label_0 = [0 for i in range(38)]
# label_1 = [1 for i in range(38, len(label_test))]
#
# order = order_1 + order_0
# label = label_1 + label_0
# print(order)
# print(len(order_0), len(order_1))
# print(len(order), len(label_test))
# print(df_norm.columns.tolist())
# df_norm = df_norm[order]
# print(df_norm.columns.tolist())
feature_select = np.load('./heatmap/lge+_vs_lge-/feature_select.npy')
# heatmap(df_norm, xlabel=label, ylabel=feature_select)
# print('lge+_vs_lge-', feature_select)
print('feature_select', feature_select)