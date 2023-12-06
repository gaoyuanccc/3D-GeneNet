
import pandas as pd
import numpy as np
import numpy
# 进行交互频率转换
def multi_read_interaction_matrix(interaction_matrix_filename):
    bin_interaction = {}
    data = pd.read_csv(interaction_matrix_filename, sep='\t', header=None)

    if data.shape[1] > 3:
        # 如果数据格式不同，则以不同方式处理
        f1 = open(interaction_matrix_filename)
        f2 = f1.readlines()
        data_list = []
        for i in f2:
            i1 = i.strip('\n').strip('\t').split('\t', -1)
            data_list.append(i1)
        data_matrix = pd.DataFrame(np.array(data_list))
        m = data_matrix.shape[0]
        bin_number = []
        for d in range(m):
            bin_number.append(d)
        bin_bin_name = []
        for m_list in range(len(bin_number)):
            m_list2 = m_list
            while m_list2 < len(bin_number):
                m_list3 = '%s,%s' % (bin_number[m_list], bin_number[m_list2])
                bin_bin_name.append(m_list3)
                m_list2 = m_list2 + 1
        for bin_bin_name_list in bin_bin_name:
            bin_bin_name_list1 = bin_bin_name_list.split(',', -1)
            bin_interaction[bin_bin_name_list] = data_matrix.iloc[int(bin_bin_name_list1[0]), int(bin_bin_name_list1[1])]
    else:
        # 默认数据处理方式
        sub = ['bin1', 'bin2', 'interaction']
        data.columns = sub
        bin_interaction = dict(
            [(str(i) + ',' + str(a), str(b)) for i, a, b in zip(data['bin1'], data['bin2'], data['interaction'])])

    return bin_interaction

def multi_bin_interaction_transiform_gene_interaction( gene_n,gene1_bin,interaction_matrix_filename,gene_gene_interaction_name_np):
    """
将bin之间的互作转换为基因间的互作
    :param gene_location_np1: 基因的位置信息的字典
    :param gene_name_np: 基因的名字的形参
    :param gene_gene_interaction_name_np: 互作基因名的列表形参
    :param interaction_matrix_filename: 标准化的bin之间的互作频率文件
    :param resolution_fp: 转换频率的分辨率
    bin_begin_number: 初始bin的序号
    :return: gene_interaction_sort: 基因间的互作频率，从小到大排列；     gene_length: 基因和其对应的长度；    gene1_bin: 基因和对应的bin序号；     gene_interaction: 基因对和对应的互作频率
    """
    ####导入互作数据，得到互作bin序号和互作频率的字典
    data = pd.read_csv(interaction_matrix_filename, sep='\t', header=None)
    bin_interaction = {}
    if data.shape[1] > 3:
        f1 = open(interaction_matrix_filename)
        f2 = f1.readlines()
        data_list = []
        for i in f2:
            i1 = i.strip('\n').strip('\t').split('\t', -1)
            data_list.append(i1)
        data_matrix = pd.DataFrame(np.array(data_list))
        m = data_matrix.shape[0]
        bin_number = []
        for d in range(m):
            bin_number.append(d)
        bin_bin_name = []
        for m_list in range(len(bin_number)):
            m_list2 = m_list
            while m_list2 < len(bin_number):
                m_list3 = '%s,%s' % (bin_number[m_list], bin_number[m_list2])
                bin_bin_name.append(m_list3)
                m_list2 = m_list2 + 1
        for bin_bin_name_list in bin_bin_name:
            bin_bin_name_list1 = bin_bin_name_list.split(',', -1)
            bin_interaction[bin_bin_name_list] = data_matrix.iloc[int(bin_bin_name_list1[0]), int(bin_bin_name_list1[1])]
    else:
        data = pd.read_csv(interaction_matrix_filename, sep='\t', header=None)
        sub = ['bin1', 'bin2', 'interaction']  # 赋予列的名字
        data.columns = sub
        # df = data.set_index(['bin1','bin2'])
        # 转为字典
        bin_interaction = dict(
            [(str(i) + ',' + str(a), str(b)) for i, a, b in zip(data['bin1'], data['bin2'], data['interaction'])])

    from decimal import Decimal
    ####得到有互作信息基因对和其对应的互作频率值
    gene_remove = 0  # 计算删除的基因对(没有互作信息的)的个数，有校准的作用
    gene_interaction = {}  # 是基因对和其对应的互作频率值
    for d1 in gene_gene_interaction_name_np:
        # gene_gene_interaction_name: 一个互作基因名的列表(不包括自己和自己互作的基因对)列表
        # gene_n : 是基因和其对应的一个有效bin和多个有效bin的字典(1是一个bin，2是多个bin)
        # gene1_bin :是基因和其对应的有效bin的序号列表的字典
        # bin_interaction: 是互作的bin序号和对应的互作频率字典，用来判断有没有互作信息
        # gene_remove : 输出文件，是删除的没有互作信息的基因对的个数
        # gene_interaction : 输出文件，是有互作信息基因对和其对应的互作频率值
        (d2, d3) = d1.split(',', 1)  # 得到互作的基因对名称，gene1是d2,gene2是d3
        gene1_n = gene_n[d2]  # 这是得到gene1是一个bin还是多个bin
        gene2_m = gene_n[d3]  # 这是得到gene2是一个bin还是多个bin
        if gene1_n == 1 and gene2_m == 1:  # 当gene1和gene2都是一个bin的时候
            bin_bin1 = gene1_bin[d2][0] + ',' + gene1_bin[d3][0]  # 这是gene1和gene2所对应的bin的组合
            bin_bin2 = gene1_bin[d3][0] + ',' + gene1_bin[d2][0]  # 这是防止有反着的bin对没被找到
            if bin_bin1 in bin_interaction.keys() or bin_bin2 in bin_interaction.keys():  # 当互作的bin有互作信息时
                if bin_bin1 in bin_interaction.keys():
                    gene_interaction[d1] = float(bin_interaction[bin_bin1])  # 存入字典中，基因1和基因2的互作频率,主要是不知道是正着的有互作信息还是反着的有互作信息
                else:
                    gene_interaction[d1] = float(bin_interaction[bin_bin2])
            else:
                gene_remove = gene_remove + 1  # 不存在的话，去除的基因对加一
        else:  # 当gene1和gene2至少有一个不是一个bin的时候，就是一对多
            bin1_bin2_interaction = []  # gene1和gene2所有有互作信息的bin组合对应的互作频率
            for d4 in gene1_bin[d2]:  # 遍历gene1的bin序号
                for d5 in gene1_bin[d3]:  # 遍历gene2的bin序号
                    d6 = str(d4) + ',' + str(d5)  # 然后把bin组合起来
                    d7 = str(d5) + ',' + str(d4)  # 同样的，防止反着的bin互作没找到
                    if d6 in bin_interaction.keys() or d7 in bin_interaction.keys():  # 当互作的bin有互作信息时
                        if d6 in bin_interaction.keys():  # 存入bin组合对应的互作频率列表中,主要是不知道是正着的有互作信息还是反着的有互作信息
                            bin1_bin2_interaction.append(float(bin_interaction[d6]))  # 存入的是互作频率
                        else:
                            bin1_bin2_interaction.append(float(bin_interaction[d7]))
            if len(bin1_bin2_interaction) == 0:  # 当有互作信息的互作bin个数是0时
                gene_remove = gene_remove + 1  # 删除基因对个数加一
            else:
                gene_interaction[d1] = numpy.mean(bin1_bin2_interaction)  # 存入字典中，基因对对应的是有互作信息的互作频率的平均值
    gene_interaction_decimal = {}
    for g_i_key,g_i_value in gene_interaction.items():
        interaction_d = '%.6f' % g_i_value
        gene_interaction_decimal[g_i_key] = float(interaction_d)
        ####把基因对按照互作频率的大小从小到大进行排序,保存为列表

    return gene_interaction,gene_interaction_decimal

def multi_remove_distance_filter(gene_interaction_np, gene_distance_np):
    gene_distance_new = {}
    for i in gene_interaction_np.keys():
        if i in gene_distance_np.keys():
            gene_distance_new[i] = gene_distance_np[i]
        else:
            gene_distance_new[i] = '-'
    return gene_distance_new

def multi_sort_and_combine_dicts(interaction_dict, distance_dict):
    # 排序字典
    interaction_asc = dict(sorted(interaction_dict.items(), key=lambda x: x[1]))

    # 组合字典
    interaction_distance_intra_asc = {key: [interaction_dict[key], distance_dict[key]]
                                      for key in interaction_asc}


    return interaction_distance_intra_asc


