
import pandas as pd
import numpy as np

from scipy import stats

def multi_read_cid_interaction_file(cid_interaction_file):
    data = pd.read_csv(cid_interaction_file, sep='\t', header=None, names=['bin1', 'bin2', 'interaction'])

    # 从数据框中获取每一行，用整数型的bin编号作为字典的键，交互频率作为值
    interaction_dict = {(int(row[0]), int(row[1])): row[2] for _, row in data.iterrows()}

    # 添加反向交互到字典中，即(bin2, bin1)的交互频率也被包括在内
    interaction_dict.update({(k[1], k[0]): v for k, v in interaction_dict.items()})

    # 使用log2转换交互频率
    interaction_dict_log = {k: np.log2(v + 1) for k, v in interaction_dict.items()}

    return  interaction_dict_log

def multi_identify_cid_1(chr_length,cid_resolution,bin_begin_number,interaction_dict_log,gene_location):
    if (chr_length % cid_resolution) == 0:  # 如果没有余值，则不需要除以的商加1
        bin_count_bed = (chr_length // cid_resolution)
    else:
        bin_count_bed = (chr_length // cid_resolution) + 1
    bin_name_bed = []  # 获取bin的序号列表
    for bin_count_bed_list in range(bin_begin_number, bin_count_bed + bin_begin_number):  # bin_number_begin的初始值是0
        bin_name_bed.append(bin_count_bed_list)

    bin_location = {}
    bin_bed_begin = 0
    bin_bed_finna = cid_resolution
    for bin_name_bed_list in bin_name_bed:
        bin_location[str(bin_name_bed_list)] = [bin_bed_begin, bin_bed_finna]
        bin_bed_begin = bin_bed_begin + cid_resolution
        bin_bed_finna = bin_bed_finna + cid_resolution

    # 创建一个新的字典来保存每个bin左右两边各10个的bin序号，注意这里不包括bin自己和自己的互作序号
    bin_sides_corrected = {}

    # 为每个bin获取其左右两边各10个的bin序号，注意这里的索引需要进行适当的调整
    num_bins = bin_count_bed
    for i in range(1, num_bins + 1):
        left_bins = [(i - j) % num_bins if (i - j) % num_bins != 0 else num_bins for j in range(1, 11)]
        right_bins = [(i + j) % num_bins if (i + j) % num_bins != 0 else num_bins for j in range(1, 11)]

        bin_sides_corrected[i] = {'left': [(i, j) for j in left_bins], 'right': [(i, j) for j in right_bins]}


    # 创建一个新的字典来保存每个bin左右两边各10个的交互频率
    bin_interaction_sides = {}

    # 为每个bin获取其左右两边各10个的交互频率
    for bin, sides in bin_sides_corrected.items():
        left_interactions = [interaction_dict_log.get(pair, 0) for pair in sides['left']]
        right_interactions = [interaction_dict_log.get(pair, 0) for pair in sides['right']]

        bin_interaction_sides[bin] = {'left': left_interactions, 'right': right_interactions}

    # 创建一个新的字典来保存每个bin的交互偏好性方向
    bin_interaction_preference_paired = {}

    # 为每个bin通过配对样本t检验（单尾）来确定其交互偏好性方向
    for bin, sides in bin_interaction_sides.items():
        left_interactions = sides['left']
        right_interactions = sides['right']

        # 进行配对样本t检验
        t_stat, p_val = stats.ttest_rel(right_interactions, left_interactions)

        # 由于我们进行的是单尾检验（并且我们关心的是right_interactions是否显著大于left_interactions），我们需要将p值除以2
        p_val /= 2

        bin_interaction_preference_paired[bin] = {'t_stat': t_stat, 'p_val': p_val}

    # Identify significantly interacting bins
    significant_bins = [bin for bin, pref in bin_interaction_preference_paired.items() if abs(pref['t_stat']) > 1.81]

    # Separate bins that are significantly interacting to the right and to the left
    bins_significant_right = [bin for bin, pref in bin_interaction_preference_paired.items() if pref['t_stat'] > 1.81]
    bins_significant_left = [bin for bin, pref in bin_interaction_preference_paired.items() if pref['t_stat'] < -1.81]

    # Negative for bins significantly interacting to the left
    bins_significant_signed = [bin if bin in bins_significant_right else -bin for bin in significant_bins]

    # 初始化一个空列表来存储CID
    CIDs_corrected = []
    CID_boundary = []
    # 从第一个显著互作的bin开始
    current_bin_index = 0

    # 如果当前的bin不是最后一个bin
    while current_bin_index < len(bins_significant_signed):
        # 检查当前的bin是否显著向右互作
        if bins_significant_signed[current_bin_index] > 0:
            # 如果当前的bin显著向右互作，那么它就是一个新CID的开始
            start_bin = bins_significant_signed[current_bin_index]

            # 继续检查下一个bin，直到找到一个bin显著向左互作
            while current_bin_index < len(bins_significant_signed) and bins_significant_signed[current_bin_index] > 0:
                current_bin_index += 1

            # 找到显著向左互作的bin，然后继续检查下一个bin，直到找到一个bin显著向右互作
            while current_bin_index < len(bins_significant_signed) and bins_significant_signed[current_bin_index] < 0:
                current_bin_index += 1

            # 找到显著向右互作的bin，它是当前CID的结束
            if current_bin_index < len(bins_significant_signed):
                end_bin = abs(bins_significant_signed[current_bin_index]) - 2
            else:  # 如果已经超出范围，使用最后一个bin作为结束
                end_bin = abs(bins_significant_signed[-1])

            # 将当前的CID加入到列表中
            CIDs_corrected.append((start_bin, end_bin))
            current_bin_index -= 1
            CID_boundary.append( end_bin + 1)

        # 移动到下一个bin
        current_bin_index += 1


    if bins_significant_signed[0] > 0 and bins_significant_signed[-1] > 0:
        CIDs_corrected[0] = (CIDs_corrected[-1][0], CIDs_corrected[0][1])
        CIDs_corrected.pop(-1)
        CID_boundary.pop(-1)

    else:
        # 修正最后一个CID的结束bin为第一个CID的开始bin序号-2
        CIDs_corrected[-1] = (CIDs_corrected[-1][0], CIDs_corrected[0][0] - 2)
        CID_boundary.pop(-1)
        CID_boundary.insert(0, CIDs_corrected[0][0] - 1)

    # 创建一个字典，键为CID的索引（从1开始），值为对应的bin范围
    CID_range = {}
    for i in range(len(CIDs_corrected)):
        CID_range[i + 1] = CIDs_corrected[i]

    # Create a dictionary to map bin index to the range of genomic positions
    bin_index_dict = {i: [(i - 1) * cid_resolution, i * cid_resolution] for i in range(1, num_bins + 1)}

    # 然后遍历CID范围
    CID_range_genome = {}
    for i, j in CID_range.items():
        range_genome = []
        range_genome.append(bin_index_dict[j[0]][0])
        range_genome.append(bin_index_dict[j[1]][1])
        CID_range_genome[i] = range_genome

    CID_gene = {}
    # 然后遍历CID，再遍历基因
    for i, j in CID_range_genome.items():
        gene_list = []
        if j[0] < j[1]:
            for x in gene_location.keys():
                if j[0] <= int(gene_location[x][0]) and int(gene_location[x][1]) <= j[1]:
                    gene_list.append(x)
        else:
            for x in gene_location.keys():
                if j[0] <= int(gene_location[x][0]) or int(gene_location[x][1]) <= j[1]:
                    gene_list.append(x)
        CID_gene[i] = gene_list

    gene_cid = {}
    gene_list_cid = []
    for i, j in CID_gene.items():
        for x in j:
            gene_cid[x] = i
            gene_list_cid.append(x)

    gene_list_boundary = []
    for i in gene_location.keys():
        if i not in gene_list_cid:
            gene_list_boundary.append(i)
            gene_cid[i] = 'boundary'

    return  gene_cid,gene_list_cid,gene_list_boundary,CID_gene,CIDs_corrected,CID_boundary,bin_interaction_preference_paired,bin_name_bed[-1]

def multi_identify_cid_after(chr_length, cid_resolution, bin_begin_number,interaction_dict_log,bin_chr_end,gene_location):
    if (chr_length % cid_resolution) == 0:  # 如果没有余值，则不需要除以的商加1
        bin_count_bed = (chr_length // cid_resolution)
    else:
        bin_count_bed = (chr_length // cid_resolution) + 1
    bin_name_bed = []  # 获取bin的序号列表
    for bin_count_bed_list in range(bin_begin_number, bin_count_bed + bin_begin_number):  # bin_number_begin的初始值是0
        bin_name_bed.append(bin_count_bed_list)

    bin_location = {}
    bin_bed_begin = 0
    bin_bed_finna = cid_resolution
    for bin_name_bed_list in bin_name_bed:
        bin_location[str(bin_name_bed_list)] = [bin_bed_begin, bin_bed_finna]
        bin_bed_begin = bin_bed_begin + cid_resolution
        bin_bed_finna = bin_bed_finna + cid_resolution

    # 创建一个新的字典来保存每个bin左右两边各10个的bin序号，注意这里不包括bin自己和自己的互作序号

    bin_sides_corrected = {}
    # 为每个bin获取其左右两边各10个的bin序号，注意这里的索引需要进行适当的调整
    # 创建一个加上上一个染色体最终bin序号的互作字典
    bin_sides_corrected_new = {}
    num_bins = bin_count_bed
    for i in range(1, num_bins + 1):
        left_bins = [(i - j) % num_bins if (i - j) % num_bins != 0 else num_bins for j in range(1, 11)]
        right_bins = [(i + j) % num_bins if (i + j) % num_bins != 0 else num_bins for j in range(1, 11)]

        bin_sides_corrected[i] = {'left': [(i, j) for j in left_bins], 'right': [(i, j) for j in right_bins]}
        bin_sides_corrected_new[i] = {'left': [(i + bin_chr_end, j + bin_chr_end) for j in left_bins],
                                      'right': [(i + bin_chr_end, j + bin_chr_end
                                                 ) for j in right_bins]}

    # 创建一个新的字典来保存每个bin左右两边各10个的交互频率
    bin_interaction_sides = {}

    # 为每个bin获取其左右两边各10个的交互频率
    for bin, sides in bin_sides_corrected_new.items():
        left_interactions = [interaction_dict_log.get(pair, 0) for pair in sides['left']]
        right_interactions = [interaction_dict_log.get(pair, 0) for pair in sides['right']]

        bin_interaction_sides[bin] = {'left': left_interactions, 'right': right_interactions}

    # 创建一个新的字典来保存每个bin的交互偏好性方向
    bin_interaction_preference_paired = {}

    # 为每个bin通过配对样本t检验（单尾）来确定其交互偏好性方向
    for bin, sides in bin_interaction_sides.items():
        left_interactions = sides['left']
        right_interactions = sides['right']

        # 进行配对样本t检验
        t_stat, p_val = stats.ttest_rel(right_interactions, left_interactions)

        # 由于我们进行的是单尾检验（并且我们关心的是right_interactions是否显著大于left_interactions），我们需要将p值除以2
        p_val /= 2

        bin_interaction_preference_paired[bin] = {'t_stat': t_stat, 'p_val': p_val}

    # Identify significantly interacting bins
    significant_bins = [bin for bin, pref in bin_interaction_preference_paired.items() if abs(pref['t_stat']) > 1.81]

    # Separate bins that are significantly interacting to the right and to the left
    bins_significant_right = [bin for bin, pref in bin_interaction_preference_paired.items() if pref['t_stat'] > 1.81]
    bins_significant_left = [bin for bin, pref in bin_interaction_preference_paired.items() if pref['t_stat'] < -1.81]

    # Negative for bins significantly interacting to the left
    bins_significant_signed = [bin if bin in bins_significant_right else -bin for bin in significant_bins]

    # 初始化一个空列表来存储CID
    CIDs_corrected = []
    CID_boundary = []
    # 从第一个显著互作的bin开始
    current_bin_index = 0

    # 如果当前的bin不是最后一个bin
    while current_bin_index < len(bins_significant_signed):
        # 检查当前的bin是否显著向右互作
        if bins_significant_signed[current_bin_index] > 0:
            # 如果当前的bin显著向右互作，那么它就是一个新CID的开始
            start_bin = bins_significant_signed[current_bin_index]

            # 继续检查下一个bin，直到找到一个bin显著向左互作
            while current_bin_index < len(bins_significant_signed) and bins_significant_signed[current_bin_index] > 0:
                current_bin_index += 1

            # 找到显著向左互作的bin，然后继续检查下一个bin，直到找到一个bin显著向右互作
            while current_bin_index < len(bins_significant_signed) and bins_significant_signed[current_bin_index] < 0:
                current_bin_index += 1

            # 找到显著向右互作的bin，它是当前CID的结束
            if current_bin_index < len(bins_significant_signed):
                end_bin = abs(bins_significant_signed[current_bin_index]) - 2
            else:  # 如果已经超出范围，使用最后一个bin作为结束
                end_bin = abs(bins_significant_signed[-1])

            # 将当前的CID加入到列表中
            CIDs_corrected.append((start_bin, end_bin))
            current_bin_index -= 1
            CID_boundary.append(end_bin + 1)

        # 移动到下一个bin
        current_bin_index += 1

    if bins_significant_signed[0] > 0 and bins_significant_signed[-1] > 0:
        CIDs_corrected[0] = (CIDs_corrected[-1][0], CIDs_corrected[0][1])
        CIDs_corrected.pop(-1)
        CID_boundary.pop(-1)

    else:
        # 修正最后一个CID的结束bin为第一个CID的开始bin序号-2
        CIDs_corrected[-1] = (CIDs_corrected[-1][0], CIDs_corrected[0][0] - 2)
        CID_boundary.pop(-1)
        CID_boundary.insert(0, CIDs_corrected[0][0] - 1)

    # 创建一个字典，键为CID的索引（从1开始），值为对应的bin范围
    CID_range = {}
    for i in range(len(CIDs_corrected)):
        CID_range[i + 1] = CIDs_corrected[i]

    # Create a dictionary to map bin index to the range of genomic positions
    bin_index_dict = {i: [(i - 1) * cid_resolution, i * cid_resolution] for i in range(1, num_bins + 1)}

    # 然后遍历CID范围
    CID_range_genome = {}
    for i, j in CID_range.items():
        range_genome = []
        range_genome.append(bin_index_dict[j[0]][0])
        range_genome.append(bin_index_dict[j[1]][1])
        CID_range_genome[i] = range_genome

    CID_gene = {}
    # 然后遍历CID，再遍历基因
    for i, j in CID_range_genome.items():
        gene_list = []
        if j[0] < j[1]:
            for x in gene_location.keys():
                if j[0] <= int(gene_location[x][0]) and int(gene_location[x][1]) <= j[1]:
                    gene_list.append(x)
        else:
            for x in gene_location.keys():
                if j[0] <= int(gene_location[x][0]) or int(gene_location[x][1]) <= j[1]:
                    gene_list.append(x)
        CID_gene[i] = gene_list

    gene_cid = {}
    gene_list_cid = []
    for i, j in CID_gene.items():
        for x in j:
            gene_cid[x] = i
            gene_list_cid.append(x)

    gene_list_boundary = []
    for i in gene_location.keys():
        if i not in gene_list_cid:
            gene_list_boundary.append(i)
            gene_cid[i] = 'boundary'

    return gene_cid, gene_list_cid, gene_list_boundary, CID_gene, CIDs_corrected, CID_boundary, bin_interaction_preference_paired,bin_name_bed[-1],bin_sides_corrected_new

