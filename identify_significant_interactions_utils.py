import pandas as pd
import numpy as np

def identify_significant_interactions(n_bootstrap, quantile_threshold,gene_interaction_distance_intra_asc,random_seed=None):

    # 将字典转换为适合创建DataFrame的格式
    data_list = []
    for key, value in gene_interaction_distance_intra_asc.items():
        node1, node2 = key.split(',')
        interaction, distance = value
        data_list.append([node1, node2, interaction, distance])

    # 创建DataFrame
    data = pd.DataFrame(data_list, columns=["node1", "node2", "interaction", "distance"])

    # 获取交互频率数据
    # 由于对数转换只对大于0的值定义，因此我们需要确认所有的值都大于0
    data['Log2(interaction + 1)'] = np.log2(data['interaction'] + 1)

    if random_seed is not None:
        np.random.seed(random_seed)

    # 这个函数从“data['Log_Interaction_Frequency']”中随机选择“n_bootstrap”个值（有放回地选择）
    bootstrap_samples = np.random.choice(data['Log2(interaction + 1)'], size=n_bootstrap)

    # Calculate the p-values for the observed interaction frequencies
    p_values = np.array([np.mean(bootstrap_samples > x) for x in data['Log2(interaction + 1)']])

    # 我们计算了自助样本的95%分位数作为显著性阈值（threshold = np.percentile(bootstrap_samples, 95)）。这个阈值用来判断哪些基因对的交互频率是显著的。如果一个基因对的对数交互频率大于这个阈值，那么我们就认为这个基因对的交互是显著的。
    threshold = np.percentile(bootstrap_samples, quantile_threshold)

    data['p_value'] = p_values
    quantile_threshold_5 = 1-(quantile_threshold/100)

    significant_gene_pairs = data[data['p_value'] < quantile_threshold_5]

    return significant_gene_pairs


#去除线性距离基因对
def remove_distance_def(significant_gene_pairs,rm_distance):
    filtered_significant_gene_pairs = significant_gene_pairs[significant_gene_pairs["distance"] != '-']

    return filtered_significant_gene_pairs[filtered_significant_gene_pairs["distance"] > rm_distance]


