from Bio import SeqIO

def read_genbank_file(genbank_filename,resolution,bin_begin_number):
    # 读取genbank文件
    recs = [rec for rec in SeqIO.parse(genbank_filename, "genbank")]
    gene_location = {}
    gene_direction = {}
    gene_locus_tag = {}  # locus_tag名对应的gene名
    join = []
    for rec in recs:  # 读取genbank文件中全是CDS的基因信息
        feats = [feat for feat in rec.features if feat.type == "CDS"]
        for feat in feats:
            y1 = []
            if len(feat.location.parts) == 1:
                y1.append(str(feat.location.start).replace('ExactPosition(', '').replace(')', ''))
                y1.append(str(feat.location.end).replace('ExactPosition(', '').replace(')', ''))
                gene_location[feat.qualifiers['locus_tag'][0]] = y1
                gene_direction[feat.qualifiers['locus_tag'][0]] = feat.location.strand
                if 'gene' in feat.qualifiers:
                    gene_locus_tag[feat.qualifiers['locus_tag'][0]] = feat.qualifiers['gene'][0]
            else:
                join.append(feat.location)
    genome_length = len(rec.seq)
    # 首先创建一个包含所有基因名的列表

    gene_name = []
    for l2_list2 in gene_location.keys():
        gene_name.append(l2_list2)
    # 然后创建一个包含所有基因对的列表，其中不包含重复的基因对（例如，'gene1,gene2' 和 'gene2,gene1' 被视为相同的基因对，并且只包含其中一个）
    gene_gene_interaction_name = []
    for i_gene_name in range(len(gene_name)):
        i_gene_name_2 = i_gene_name + 1  # 保证比i_gene_name大一个，防止有一样的互作名
        while i_gene_name_2 < len(gene_name):
            i_gene_name_3 = '%s,%s' % (gene_name[i_gene_name], gene_name[i_gene_name_2])
            gene_gene_interaction_name.append(i_gene_name_3)
            i_gene_name_2 = i_gene_name_2 + 1
    # 计算每个基因对之间的距离，并将这些距离存储在字典中
    gene_distance = {}
    for r1 in gene_gene_interaction_name:  # 首先得到基因对
        r2 = r1.split(',', 1)
        if float(gene_location[r2[0]][1]) < float(gene_location[r2[1]][0]):  # 如果gene1在gene2的前面的话
            r3 = float(gene_location[r2[1]][0]) - float(gene_location[r2[0]][1])  # gene1和gene2之间的距离
            if r3 <= genome_length // 2:  # 如果基因之间的距离小于一半
                gene_distance[r1] = r3  # 那么基因间的距离就是r3
            else:  # 如果基因的距离大于一半
                r4 = genome_length - float(gene_location[r2[1]][1])  # 这是gene2的末尾位置到基因末尾的长度
                r5 = r4 + float(gene_location[r2[0]][0])  # 基因间的长度就等于gene1的初始位置加上gene2的末尾位置到基因末尾的长度
                gene_distance[r1] = r5
        elif float(gene_location[r2[1]][1]) < float(gene_location[r2[0]][0]):  # 如果gene1在gene2的后面的话
            r6 = float(gene_location[r2[0]][0]) - float(gene_location[r2[1]][1])  # gene2和gene1之间的距离
            if r6 <= genome_length // 2:  # 如果基因之间的距离小于一半
                gene_distance[r1] = r6  # 那么基因间的距离就是r6
            else:  # 如果基因间的距离大于一半
                r7 = genome_length - float(gene_location[r2[0]][1])
                r8 = r7 + float(gene_location[r2[1]][0])
                gene_distance[r1] = r8
        else:  # 除了上面两种，其他的就是gene1和gene2有交集，所以距离全都归为0
            gene_distance[r1] = 0
    # 把基因对按照线性距离的大小从大到小进行排序,保存为列表

    gene_distance_sort = sorted(gene_distance.items(), key=lambda x: x[1], reverse=True)  # 已经是列表了

    if (genome_length % resolution) == 0:  # 如果没有余值，则不需要除以的商加1
        bin_count_bed = (genome_length // resolution)
    else:
        bin_count_bed = (genome_length // resolution) + 1
    bin_name_bed = []  # 获取bin的序号列表
    for bin_count_bed_list in range(bin_begin_number, bin_count_bed + bin_begin_number):  # bin_number_begin的初始值是0
        bin_name_bed.append(bin_count_bed_list)

    bin_location = {}
    bin_bed_begin = 0
    bin_bed_finna = resolution
    for bin_name_bed_list in bin_name_bed:
        bin_location[str(bin_name_bed_list)] = [bin_bed_begin, bin_bed_finna]
        bin_bed_begin = bin_bed_begin + resolution
        bin_bed_finna = bin_bed_finna + resolution

    return gene_location, gene_direction, gene_locus_tag, genome_length, join,gene_name, gene_gene_interaction_name,gene_distance,bin_name_bed,bin_location
