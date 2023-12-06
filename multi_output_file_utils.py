

import pandas as pd

import matplotlib.pyplot as plt
def multi_save_cid_range_list(cid_list, output_cid_range_path):
    """
    将CID列表保存到文件中。

    :param cid_list: CID范围的列表。
    :param output_path: 输出文件的路径。
    """
    # 创建一个新的列表，包含CID序号和范围
    cid_data = [("CID" + str(index + 1), range_tuple) for index, range_tuple in enumerate(cid_list)]

    # 转换为DataFrame
    df = pd.DataFrame(cid_data, columns=["CID number", "CID range"])

    # 保存到文件
    df.to_csv(output_cid_range_path, index=False)


def multi_save_dict_to_file(data_dict, output_path):
    """
    将字典保存到文件中。

    :param data_dict: 要保存的字典。
    :param output_path: 输出文件的路径。
    """
    # 转换字典为列表
    data_list = [(bin, data['t_stat'], data['p_val']) for bin, data in data_dict.items()]

    # 转换为DataFrame
    df = pd.DataFrame(data_list, columns=["bin", "t_stat", "p_val"])

    # 保存到文件
    df.to_csv(output_path, index=False)

def multi_save_gene_cid_to_file(data_dict, output_path):
    """
    将字典保存到文件中。

    :param data_dict: 要保存的字典。
    :param output_path: 输出文件的路径。
    """
    # 转换字典为列表
    data_list = [(bin, data) for bin, data in data_dict.items()]

    # 转换为DataFrame
    df = pd.DataFrame(data_list, columns=["gene", "cid"])

    # 保存到文件
    df.to_csv(output_path, index=False)

def multi_save_dataframe_to_file(data_dataframe, output_path):
    """
    将字典保存到文件中。

    :param data_dataframe: 要保存的数据框
    :param output_path: 输出文件的路径。
    """

    # 保存到文件
    data_dataframe.to_csv(output_path, index=False)


def multi_plot_cid_interaction_file(bin_interaction_preference_paired,CID_boundary,output_path):
    plt.rcParams["font.size"] = 24
    plt.figure(figsize=(15, 10))

    bins = [bin for bin in bin_interaction_preference_paired.keys()]
    t_values = [pref['t_stat'] for pref in bin_interaction_preference_paired.values()]
    colors = ['#81b29a' if t > 0 else '#6d597a' for t in t_values]

    plt.bar(bins, t_values, color=colors, width=1.3)
    plt.xlabel('Bin Number')
    plt.ylabel('t value')

    # Adjust y-axis to remove gap at y=0
    plt.ylim(min(t_values)-1, max(t_values)+ 1)

    # Adjust x-axis to remove gap
    plt.xlim(min(bins) - 2, max(bins))

    plt.axhline(y=1.81, color='k', linestyle='--')
    plt.axhline(y=-1.81, color='k', linestyle='--')

    for boundary in CID_boundary:
        plt.axvline(x=boundary, color='k', linestyle='--', ymax=0.565)

    plt.xticks(CID_boundary, rotation=90, fontsize=24)
    plt.yticks([0, 1.81, -1.81], fontsize=24)
    # 保存图像
    plt.savefig(output_path, format='pdf')
