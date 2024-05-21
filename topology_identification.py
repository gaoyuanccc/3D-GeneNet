import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler


def process_interaction_gene_pairs(interaction_gene_pairs):
    # 构建网络
    G = nx.from_pandas_edgelist(interaction_gene_pairs, 'node1', 'node2', create_using=nx.Graph())

    # Calculate degree centrality, closeness centrality, and betweenness centrality for distance-based network
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # region 添加新的一列
    interaction_gene_pairs['node1_betweenness_centrality'] = interaction_gene_pairs['node1'].map(betweenness_centrality)
    interaction_gene_pairs['node2_betweenness_centrality'] = interaction_gene_pairs['node2'].map(betweenness_centrality)

    interaction_gene_pairs['node1_degree_centrality'] = interaction_gene_pairs['node1'].map(degree_centrality)
    interaction_gene_pairs['node2_degree_centrality'] = interaction_gene_pairs['node2'].map(degree_centrality)

    # region 计算求和
    interaction_gene_pairs['betweenness_centrality_sum'] = interaction_gene_pairs['node1_betweenness_centrality'] + \
                                                           interaction_gene_pairs['node2_betweenness_centrality']
    interaction_gene_pairs['degree_centrality_sum'] = interaction_gene_pairs['node1_degree_centrality'] + \
                                                      interaction_gene_pairs['node2_degree_centrality']
    # endregion

    # region 0-1标准化
    # 初始化MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 选择特定列进行标准化
    columns_to_scale = ['betweenness_centrality_sum']
    # 对选定的列进行标准化
    scaled_values = scaler.fit_transform(interaction_gene_pairs[columns_to_scale])
    # 将标准化后的数据转换回DataFrame格式，方便查看
    interaction_gene_pairs['betweenness_centrality_sum_normalized'] = scaled_values[:, 0]  # 新列存放Indicator1的标准化数据
    # endregion

    # region 0-1标准化
    # 初始化MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 选择特定列进行标准化
    columns_to_scale = ['degree_centrality_sum']
    # 对选定的列进行标准化
    scaled_values = scaler.fit_transform(interaction_gene_pairs[columns_to_scale])
    # 将标准化后的数据转换回DataFrame格式，方便查看
    interaction_gene_pairs['degree_centrality_sum_normalized'] = scaled_values[:, 0]  # 新列存放Indicator1的标准化数据
    # endregion

    # region 0-1标准化
    # 初始化MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 选择特定列进行标准化
    columns_to_scale = ['interaction']
    # 对选定的列进行标准化
    scaled_values = scaler.fit_transform(interaction_gene_pairs[columns_to_scale])
    # 将标准化后的数据转换回DataFrame格式，方便查看
    interaction_gene_pairs['interaction_normalized'] = scaled_values[:, 0]  # 新列存放Indicator1的标准化数据
    # endregion

    interaction_gene_pairs['SSC score'] = (1 - interaction_gene_pairs['betweenness_centrality_sum_normalized']) * 0.3 + (
                1 - interaction_gene_pairs['degree_centrality_sum_normalized']) * 0.3 + interaction_gene_pairs[
                                          'interaction_normalized'] * 0.4

    interaction_gene_pairs = interaction_gene_pairs.drop('node1_betweenness_centrality', axis=1)
    interaction_gene_pairs = interaction_gene_pairs.drop('node2_betweenness_centrality', axis=1)
    interaction_gene_pairs = interaction_gene_pairs.drop('node1_degree_centrality', axis=1)
    interaction_gene_pairs = interaction_gene_pairs.drop('node2_degree_centrality', axis=1)
    interaction_gene_pairs = interaction_gene_pairs.drop('betweenness_centrality_sum', axis=1)
    interaction_gene_pairs = interaction_gene_pairs.drop('degree_centrality_sum', axis=1)

    return interaction_gene_pairs
