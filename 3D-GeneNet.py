
import argparse
import os
import time
import sys
from genbank_utils import read_genbank_file
from interaction_utils import read_interaction_matrix, calculate_gene_interaction,remove_distance_filter,sort_and_combine_dicts
from identify_significant_interactions_utils import identify_significant_interactions,remove_distance_def
from identify_cid_utils import identify_cid
from output_file_utils import save_cid_range_list,save_dict_to_file,save_dataframe_to_file,save_gene_cid_to_file,plot_cid_interaction_file
from multi_genbank_utils import multi_gene_location
from multi_interaction_utils import multi_bin_interaction_transiform_gene_interaction,multi_remove_distance_filter,multi_read_interaction_matrix,multi_sort_and_combine_dicts
from multi_identify_significant_interactions_utils import multi_remove_distance,multi_identify_significant_interactions
from multi_identify_cid_utils import multi_identify_cid_after,multi_identify_cid_1,multi_read_cid_interaction_file
from multi_output_file_utils import multi_save_dict_to_file,multi_save_dataframe_to_file,multi_save_gene_cid_to_file,multi_save_cid_range_list,multi_plot_cid_interaction_file

import numpy as np
start = time.time()

def parse_args():
    parser = argparse.ArgumentParser(description='Gene association network obtained by three-dimensional spatial interaction frequency of bacterial chromosomes.')
    parser.add_argument('-i', '--input_file_path', type=str, help='Path to the input file directory.')
    parser.add_argument('-gb','--genbank_file', type=str, help='Name of the Genbank file.')
    parser.add_argument('-if','--interaction_file', type=str, help='Name of the interaction frequency file.')
    parser.add_argument('-ic','--cid_interaction_file', type=str, help='Name of the CID interaction frequency file for identification.',default=None)
    parser.add_argument('-o', '--outfile_path', type=str, help='Path to the output directory.')
    parser.add_argument('-b','--bin_begin_number', type=int, help='Initial bin number for analysis.',default=1)
    parser.add_argument('-r','--resolution', type=int, default=1000, help='Resolution parameter for analysis (default: 1000).')
    parser.add_argument('-rc', '--resolution_cid', type=int, default=10000, help='CID-specific resolution parameter (default: 10000).')
    parser.add_argument('-n',"--n_bootstrap", type=int, default=1000, help="Number of bootstrap iterations.")
    parser.add_argument('-s','--seed', type=int, help='Seed for random number generation to ensure reproducibility.', default=None)
    parser.add_argument('-q',"--quantile_threshold", type=float, default=95, help="Quantile threshold for analysis.")
    parser.add_argument('-d', '--remove_distance', type=int, help='inear distance threshold for removing gene pairs.', default=0)
    parser.add_argument('-p', '--multichromosome', type=int, default=0,help='Used to determine if the target strain is multichromosome; if the strain is multichromosome, set to 1 and is 0 by default.')
    parser.add_argument('-gbc', '--multichromosomeGenbank', type=str, default='',help='If the strain is multichromosome, enter the Genbank filenames for the chromosomes in order, separated by a "/" symbol.')

    args = parser.parse_args()

    return args
def main():
    args = parse_args()
    ####基础设置
    print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n' + 'Run 3D-GeneNet' + '\n')

    if args.multichromosome == 0: #如果不是多染色体
        # 读取和处理Genbank文件
        print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()) + '\n' + 'Convert interaction frequency' + '\n')
        os.chdir(args.input_file_path)
        gene_location, gene_direction, gene_locus_tag, genome_length, join,gene_name, gene_gene_interaction_name,gene_distance,bin_name_bed,bin_location = read_genbank_file(args.genbank_file,args.resolution,args.bin_begin_number)

        # 读取和处理交互频率文件
        bin_interaction = read_interaction_matrix(args.interaction_file)

        # 计算交互频率
        # 可能需要根据实际情况调整 calculate_gene_interaction 函数的参数
        gene_interaction,gene_interaction_decimal = calculate_gene_interaction(bin_interaction, gene_location, gene_gene_interaction_name, args.resolution, args.bin_begin_number,gene_name,bin_location)

        # 这里可以添加更多的处理逻辑或输出
        # 删除线性距离中总的交互频率没有的基因对,删除总的交互频率没有的基因对
        gene_distance_remove = remove_distance_filter(gene_interaction_decimal, gene_distance)

        gene_interaction_distance_intra_asc, gene_interaction_distance_dis_desc = sort_and_combine_dicts(gene_interaction_decimal, gene_distance_remove)

        # 如果用户提供了种子，则设置随机数种子
        if args.seed is not None:
            np.random.seed(args.seed)

        #选择基因对
        print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n' + 'Select gene pairs' + '\n')

        significant_gene_pairs = identify_significant_interactions(args.n_bootstrap,args.quantile_threshold,gene_interaction_distance_intra_asc,args.seed)
        select_gene_pairs = remove_distance_def(significant_gene_pairs,args.remove_distance)


        # 设置您的文件夹路径
        folder_path = args.outfile_path

        # 检查文件夹是否存在，如果不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        os.chdir(args.outfile_path)
        if  args.cid_interaction_file == None:
            # 保存基因对文件
            save_dataframe_to_file(select_gene_pairs, 'gene_association_network.csv')
        else:
            #识别CID
            print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n' + 'Identify CIDs' + '\n')
            os.chdir(args.input_file_path)
            CID_boundary, CIDs_corrected, bin_interaction_preference_paired,gene_cid = identify_cid(args.cid_interaction_file,gene_location,args.resolution_cid)

            #输出文件
            print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n' + 'Output files' + '\n')
            os.chdir(args.outfile_path)
            #保存CID序号和范围文件
            save_cid_range_list(CIDs_corrected, "cid_number.csv")

            #保存基因和CID序号
            save_gene_cid_to_file(gene_cid, "gene_cid.csv")

            #保存bin和对应的t值和p值
            save_dict_to_file(bin_interaction_preference_paired,'cid_t_pvalue.csv')

            #保存基因对文件
            save_dataframe_to_file(select_gene_pairs,'gene_association_network.csv')

            plot_cid_interaction_file(bin_interaction_preference_paired,CID_boundary,"cid_bin_preference.pdf")
    else:

        print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()) + '\n' + 'Convert interaction frequency' + '\n')
        os.chdir(args.input_file_path)
        gene_location, genome_length, gene_direction, gene_locus_tag, gene_name, gene_distance, gene_length, gene1_bin, gene_n, gene1_bin_merge, gene_gene_interaction_name, genome_length_chr, gene_location_chr = multi_gene_location(args.multichromosomeGenbank, args.resolution, args.bin_begin_number)

        # 读取和处理交互频率文件
        bin_interaction = multi_read_interaction_matrix(args.interaction_file)

        # 计算交互频率
        # 可能需要根据实际情况调整 calculate_gene_interaction 函数的参数
        gene_interaction, gene_interaction_decimal = multi_bin_interaction_transiform_gene_interaction(gene_n,gene1_bin_merge,args.interaction_file,gene_gene_interaction_name)

        # 这里可以添加更多的处理逻辑或输出
        # 删除线性距离中总的交互频率没有的基因对,删除总的交互频率没有的基因对
        gene_distance_remove = multi_remove_distance_filter(gene_interaction_decimal, gene_distance)

        gene_interaction_distance_intra_asc = multi_sort_and_combine_dicts(gene_interaction_decimal, gene_distance_remove)


        # 如果用户提供了种子，则设置随机数种子
        if args.seed is not None:
            np.random.seed(args.seed)
        print(
            '-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n' + 'Select gene pairs' + '\n')
        significant_gene_pairs = multi_identify_significant_interactions(args.n_bootstrap, args.quantile_threshold,
                                                                   gene_interaction_distance_intra_asc, args.seed)
        select_gene_pairs = multi_remove_distance(significant_gene_pairs, args.remove_distance)


        # 设置您的文件夹路径
        folder_path = args.outfile_path

        # 检查文件夹是否存在，如果不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        os.chdir(args.outfile_path)

        if  args.cid_interaction_file == None:
            print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n' + 'Output files' + '\n')
            # 保存基因对文件
            save_dataframe_to_file(select_gene_pairs, 'gene_association_network.csv')

        else:
            print(
                '-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '\n' + 'Identify CIDs' + '\n')
            os.chdir(args.input_file_path)
            interaction_dict_log = multi_read_cid_interaction_file(args.cid_interaction_file)
            for chr_length in genome_length_chr.keys():
                if chr_length == 0:
                    gene_cid, gene_list_cid, gene_list_boundary, CID_gene, CIDs_corrected, CID_boundary, bin_interaction_preference_paired, bin_chr1_end = multi_identify_cid_1(
                        genome_length_chr[chr_length], args.resolution_cid


                        , args.bin_begin_number, interaction_dict_log,
                        gene_location_chr[chr_length])

                    os.chdir(args.outfile_path)
                    # 保存CID序号和范围文件
                    multi_save_cid_range_list(CIDs_corrected, "chr" + str(chr_length + 1) + "_cid_number.csv")

                    # 保存基因和CID序号
                    multi_save_gene_cid_to_file(gene_cid, "chr" + str(chr_length + 1) + "_gene_cid.csv")

                    # 保存bin和对应的t值和p值
                    multi_save_dict_to_file(bin_interaction_preference_paired, "chr" + str(chr_length + 1) + '_cid_t_pvalue.csv')

                    # 保存cid中bin的偏好图
                    multi_plot_cid_interaction_file(bin_interaction_preference_paired,CID_boundary,"chr" + str(chr_length + 1) + "_bin_preference.pdf")

                else:
                    gene_cid, gene_list_cid, gene_list_boundary, CID_gene, CIDs_corrected, CID_boundary, bin_interaction_preference_paired, bin_chr1_end, bin_sides_corrected_new = multi_identify_cid_after(
                        genome_length_chr[chr_length], args.resolution_cid, args.bin_begin_number, interaction_dict_log, bin_chr1_end,
                        gene_location_chr[chr_length])
                    # 保存CID序号和范围文件

                    os.chdir(args.outfile_path)
                    print('-' * 30 + '\n' + time.strftime('%Y-%m-%d %H:%M:%S',
                                                          time.localtime()) + '\n' + 'Output files' + '\n')
                    multi_save_cid_range_list(CIDs_corrected, "chr" + str(chr_length + 1) + "_cid_number.csv")

                    # 保存基因和CID序号
                    multi_save_gene_cid_to_file(gene_cid, "chr" + str(chr_length + 1) + "_gene_cid.csv")

                    # 保存bin和对应的t值和p值
                    multi_save_dict_to_file(bin_interaction_preference_paired, "chr" + str(chr_length + 1) + '_cid_t_pvalue.csv')

                    # 保存cid中bin的偏好图
                    multi_plot_cid_interaction_file(bin_interaction_preference_paired, CID_boundary,
                                                    "chr" + str(chr_length + 1) + "_bin_preference.pdf")

            multi_save_dataframe_to_file(select_gene_pairs, 'gene_association_network.csv')






if __name__ == "__main__":
    main()

end = time.time()
print('Time:    ' + str(end - start) + 's') #整个程序运行的时间