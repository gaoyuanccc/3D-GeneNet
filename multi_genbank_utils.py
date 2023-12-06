from Bio import SeqIO

def multi_gene_location(multichromosomeGenbank,resolution,bin_begin_number):

    # 分割Genbank文件名
    genebank_name = multichromosomeGenbank.split('/', -1)  # genbank文件名列表

    def genbank_location(genbank_filename):
        """
    将输入的原核细菌的genbank文件中的基因名和位置信息提取出来
        :param genbank_filename:输入的genbank文件名
        :return: 得到的基因名和对应的位置信息的字典; 以及基因的长度还有基因的方向+1是正的，-1是反的
        """
        import os
        # 读取genbank文件
        recs = [rec for rec in SeqIO.parse(genbank_filename, "genbank")]
        for rec in recs:  # 这一步是读取genbank文件的文件中全是CDS的基因信息
            feats = [feat for feat in rec.features if feat.type == "CDS"]
        gene_location = {}
        gene_direction = {}
        gene_locus_tag = {}  # locus_tag名对应的gene名
        join = []
        for i in feats:
            y1 = []
            if len(i.location.parts) == 1:
                y1.append(str(i.location.start).replace('ExactPosition(', '').replace(')', ''))
                y1.append(str(i.location.end).replace('ExactPosition(', '').replace(')', ''))
                gene_location[i.qualifiers['locus_tag'][0]] = y1
                gene_direction[i.qualifiers['locus_tag'][0]] = i.location.strand
                if 'gene' in i.qualifiers.keys():
                    gene_locus_tag[i.qualifiers['locus_tag'][0]] = i.qualifiers['gene'][0]
            else:
                join.append(i.location)
        genome_length = len(rec.seq)

        return gene_location, genome_length, gene_direction, gene_locus_tag

    def gene_gene_distance(genome_length_fp, gene_location_np):
        """
    通过基因的位置信息和基因的长度信息，获取基因对之间的线性距离的字典,并且将基因对按照线性距离从大到小排序
        :param gene_location_np: 基因的位置信息的字典
        :param genome_length_fp: 原核基因组的长度
        :return: gene_distance_sort: 返回的是基因对和对应的线性距离从大到小排序的字典
        gene_name: 基因名的列表
        gene_gene_interaction_name: 互作的基因名，不包括自己和自己的互作的基因
        gene_distance: 互作的基因对名字和对应的线性距离
        """
        ####得到一个基因对和它们之间的线性距离的字典 基因对：线性距离
        # 先得到一个gene名的列表
        gene_name = []
        for l2_list2 in gene_location_np.keys():
            gene_name.append(l2_list2)
        # 再得到一个互作基因名的列表(不包括自己和自己互作的基因对)
        gene_gene_interaction_name = []
        for i_gene_name in range(len(gene_name)):
            i_gene_name_2 = i_gene_name + 1  # 保证比i_gene_name大一个，防止有一样的互作名
            while i_gene_name_2 < len(gene_name):
                i_gene_name_3 = '%s,%s' % (gene_name[i_gene_name], gene_name[i_gene_name_2])
                gene_gene_interaction_name.append(i_gene_name_3)
                i_gene_name_2 = i_gene_name_2 + 1
        # 得比较基因的位置，必须是位置大的末尾位置减去位置小的初始位置 基因长度：4639675
        gene_distance = {}
        for r1 in gene_gene_interaction_name:  # 首先得到基因对
            r2 = r1.split(',', 1)
            if float(gene_location_np[r2[0]][1]) < float(gene_location_np[r2[1]][0]):  # 如果gene1在gene2的前面的话
                r3 = float(gene_location_np[r2[1]][0]) - float(gene_location_np[r2[0]][1])  # gene1和gene2之间的距离
                if r3 <= genome_length_fp // 2:  # 如果基因之间的距离小于一半
                    gene_distance[r1] = r3  # 那么基因间的距离就是r3
                else:  # 如果基因的距离大于一半
                    r4 = genome_length_fp - float(gene_location_np[r2[1]][1])  # 这是gene2的末尾位置到基因末尾的长度
                    r5 = r4 + float(gene_location_np[r2[0]][0])  # 基因间的长度就等于gene1的初始位置加上gene2的末尾位置到基因末尾的长度
                    gene_distance[r1] = r5
            elif float(gene_location_np[r2[1]][1]) < float(gene_location_np[r2[0]][0]):  # 如果gene1在gene2的后面的话
                r6 = float(gene_location_np[r2[0]][0]) - float(gene_location_np[r2[1]][1])  # gene2和gene1之间的距离
                if r6 <= genome_length_fp // 2:  # 如果基因之间的距离小于一半
                    gene_distance[r1] = r6  # 那么基因间的距离就是r6
                else:  # 如果基因间的距离大于一半
                    r7 = genome_length_fp - float(gene_location_np[r2[0]][1])
                    r8 = r7 + float(gene_location_np[r2[1]][0])
                    gene_distance[r1] = r8
            else:  # 除了上面两种，其他的就是gene1和gene2有交集，所以距离全都归为0
                gene_distance[r1] = 0
        # 把基因对按照线性距离的大小从大到小进行排序,保存为列表

        gene_distance_sort = sorted(gene_distance.items(), key=lambda x: x[1], reverse=True)  # 已经是列表了
        return gene_name, gene_gene_interaction_name, gene_distance

    def bin_interaction_transiform_gene_poly(genome_length, resolution_fp, gene_location_np1, gene_name_np,bin_begin_number):
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
        ####导入备注文件，得到bin和它的位置信息的字典
        if (genome_length % resolution_fp) == 0:  # 如果没有余值，则不需要除以的商加1
            bin_count_bed = (genome_length // resolution_fp)
        else:
            bin_count_bed = (genome_length // resolution_fp) + 1
        bin_name_bed = []  # 获取bin的序号列表
        for bin_count_bed_list in range(bin_begin_number, bin_count_bed + bin_begin_number):  # bin_number_begin的初始值是0
            bin_name_bed.append(bin_count_bed_list)

        bin_location = {}
        bin_bed_begin = 0
        bin_bed_finna = resolution_fp
        for bin_name_bed_list in bin_name_bed:
            bin_location[str(bin_name_bed_list)] = [bin_bed_begin, bin_bed_finna]
            bin_bed_begin = bin_bed_begin + resolution_fp
            bin_bed_finna = bin_bed_finna + resolution_fp

            ####制作基因的长度字典 字典格式：基因名：基因长度
        gene_length = {}
        for gene_location_key in gene_location_np1.keys():  # 遍历基因名
            y2 = float(gene_location_np1[gene_location_key][1]) - float(
                gene_location_np1[gene_location_key][0])  # 基因的长度对于相应的基因的后面的位置信息减去前面的位置信息
            gene_length[gene_location_key] = y2

        ####制作基因和其对应的bin的信息的字典 字典格式：基因名：bin序号列表
        gene_bin = {}
        for b1 in gene_location_np1.keys():  # 首先遍历基因的名字
            b2 = (int(
                gene_location_np1[b1][0]) // resolution_fp) + bin_begin_number  # 基因的初始位置除于1000，整数部分就是基因的初始位置所在的bin序号
            b3 = (int(
                gene_location_np1[b1][1]) // resolution_fp) + bin_begin_number  # 基因的末尾位置除于1000，整数部分就是基因的末尾位置所在的bin序号
            b4 = []  # 通过添加每个基因相对应的序号列表
            if b2 == b3:  # 当基因的初始位置和末尾位置都位于同一个bin中时
                b4.append(str(b2))  # 那么基因对应的bin就是一个
            else:  # 当基因的初始位置和末尾位置不在同一个bin时
                b5 = int(gene_location_np1[b1][
                             1]) % resolution_fp  # 这是判断是否基因的末尾是否和bin的末尾重合了，如果重合的话，我们还是只算它在一个bin中，所以我们求的是基因的末尾位置比上1000，余数等于零就是重合了
                if b5 == 0:  # 余数等于零的话，说明正好重合了
                    while b2 <= b3 - 1:  # 重合的话，我们只算到b3的前一个，那就是b3-1
                        b4.append(str(b2))  # 这时我们只算基因的初始位置所在的bin
                        b2 = b2 + 1
                else:
                    while b2 <= b3:  # 我们依次把初始位置位于的bin序号加到末尾位置位于的bin序号就可以了
                        b4.append(str(b2))
                        b2 = b2 + 1
            gene_bin[b1] = b4  # 基因名和相对应的bin序号

        ####得到基因的性质(是一个bin还是多个bin，字典格式 gene: 1或2)和得到基因和其相对应的bin序号列表（字典格式 gene : bin序号列表)
        gene_n = {}
        gene1_bin = {}
        for gene1_name in gene_name_np:  # 获得每个基因的性质和bin序号列表的循环，得到基因的性质(是一个bin还是多个bin，字典格式 gene: 1或2)和得到基因和其相对应的bin序号列表（字典格式 gene : bin序号列表)
            # gene_name：是基因名的列表
            # bin_interaction: 是互作的bin序号和对应的互作频率字典
            # bin_location: 是bin序号对应的位置信息字典
            # gene_bin : 是基因和其对应的所位于的bin序号列表字典
            # gene_length: 是基因和对应的长度字典
            # gene_location: 是基因和其对应的位置信息列表字典
            # gene_n: 是输出的字典，是一个空字典，是基因和其对应的n值
            # gene1_bin: 是输出的字典，是一个空字典，是基因和其对应的有效bin序号列表
            n = 0
            bin_number1 = []
            if len(gene_bin[gene1_name]) == 1:  # 判断基因是否只位于一个bin内
                for a5 in gene_bin[gene1_name]:
                    bin_number1.append(a5)  # 基因所位于的bin的列表
                n = 1  # 给基因进行定性，只位于一个bin中
            else:
                if len(gene_bin[gene1_name]) == 2:  # 判断基因是否只位于两个bin内
                    o = float(gene_location_np1[gene1_name][1]) - float(
                        bin_location[gene_bin[gene1_name][1]][0])  # o是基因所位于的第二个bin的部分的长度，是由基因的末尾位置减去基因所位于的第二个bin的初始位置
                    o1 = gene_length[gene1_name] / o  # o1是基因的长度对o的倍数，是由基因的总长度比上o
                    if o1 == 2:  # 判断基因是不是被两个bin平分了
                        for a6 in gene_bin[gene1_name]:
                            bin_number1.append(a6)
                        n = 2  # 说明基因不止位于一个bin内
                    else:
                        if o1 > 2:  # 判断基因大于50%的部分
                            bin_number1.append(gene_bin[gene1_name][1])  # 如果大于2说明大于50%的部分在第二个bin
                            n = 1
                        else:
                            bin_number1.append(gene_bin[gene1_name][0])  # 如果小于2说明大于50%的部分在第一个bin
                            n = 1
                else:  # 说明基因所位于的大于等于三个bin
                    if gene_location_np1[gene1_name][0] != bin_location[gene_bin[gene1_name][0]][
                        0]:  # 基因的初始位置和第一个bin的初始位置不相同
                        if gene_location_np1[gene1_name][1] != bin_location[gene_bin[gene1_name][-1]][
                            1]:  # 判断基因的末尾位置和最后一个bin的末尾位置不相同
                            for a1 in gene_bin[gene1_name]:  # 如果基因的初始位置和末尾位置都不相同，宾序号就是bin列表去除首尾
                                bin_number1.append(a1)
                            bin_number1.pop(0)
                            bin_number1.pop()
                            if len(bin_number1) == 1:
                                n = 1
                            else:
                                n = 2
                        else:
                            for a2 in gene_bin[gene1_name]:  # bin的序号就是bin序号列表去除第一个
                                bin_number1.append(a2)
                            bin_number1.pop(0)
                            n = 2
                    else:
                        if gene_location_np1[gene1_name][1] != bin_location[gene_bin[gene1_name][-1]][
                            1]:  # 判断基因的末尾位置和最后一个bin的末尾位置不同
                            for a3 in gene_bin[gene1_name]:  # bin的序号等于bin序号列表去除最后一个
                                bin_number1.append(a3)
                            bin_number1.pop()
                            n = 2
                        else:
                            for a4 in gene_bin[gene1_name]:  # 那么bin的序号列表就是基因所占得所有bin
                                bin_number1.append(a4)
                            n = 2
            gene_n[gene1_name] = n
            gene1_bin[gene1_name] = bin_number1
        return gene_length, gene1_bin, bin_name_bed[-1], gene_n

    gene_location = {}
    genome_length = 0
    genome_length_chr = {}
    gene_direction = {}
    gene_locus_tag = {}
    gene_name = []
    gene_distance = {}
    gene_length = {}
    gene1_bin = {}
    gene_chr = {}  # 不同染色体的基因
    gene_total_bin = []  # 不同染色体的bin序号
    gene_n = {}
    gene_location_chr = {}

    def Merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res

    for i in range(len(genebank_name)):
        gene_location_sub, genome_length_sub, gene_direction_sub, gene_locus_tag_sub = genbank_location(
            genebank_name[i])
        gene_name_sub, gene_gene_interaction_name_sub, gene_distance_sub = gene_gene_distance(genome_length_sub,
                                                                                              gene_location_sub)
        if i > 0:  # 不是第一条染色体,其中bin_count是bin的数量,gene1_bin_sub中也是这条染色体上的bin的排序,这里就一直是1
            gene_length_sub, gene1_bin_sub, bin_count, gene_n_sub = bin_interaction_transiform_gene_poly(
                genome_length_sub, resolution, gene_location_sub, gene_name_sub, 1)
        else:  # 第一条染色体，其中是bin_count是第一条染色体的最后一个bin的序号，这是是bin开始的序号
            gene_length_sub, gene1_bin_sub, bin_count, gene_n_sub = bin_interaction_transiform_gene_poly(
                genome_length_sub, resolution, gene_location_sub, gene_name_sub, bin_begin_number)
        gene_location = Merge(gene_location, gene_location_sub)
        genome_length = genome_length + genome_length_sub
        gene_direction = Merge(gene_direction, gene_direction_sub)
        gene_locus_tag = Merge(gene_locus_tag, gene_locus_tag_sub)
        gene_name = gene_name + gene_name_sub
        gene_distance = Merge(gene_distance, gene_distance_sub)
        gene_length = Merge(gene_length, gene_length_sub)
        gene1_bin = Merge(gene1_bin, gene1_bin_sub)
        gene_chr[i] = gene_name_sub
        gene_total_bin.append(bin_count)
        gene_n = Merge(gene_n, gene_n_sub)
        genome_length_chr[i] = genome_length_sub
        gene_location_chr[i] = gene_location_sub
    # 计算gene1_bin，总的
    gene1_bin_merge = {}
    for j in gene_chr.keys():
        if j != 0:
            for r in gene_chr[j]:  # 遍历所有在第j条染色体的基因
                bin_list = []  # 新的bin的序列列表
                for u in gene1_bin[r]:  # 遍历第j条染色体基因的bin序号
                    for h in range(j):  # 遍历第j条染色体之前的bin序号，gene_total_bin[0]是第1条的最后bin的序号，gene_total_bin[1]是第2条的bin的数量
                        l_bin1 = int(u) + gene_total_bin[h]  # 遍历相加以后就是现在基因对应的bin的序号
                        bin_list.append(str(l_bin1))
                gene1_bin_merge[r] = bin_list
        else:
            for k in gene_chr[0]:  # 如果是在第一条染色体，直接不动，直接转换过来
                gene1_bin_merge[k] = gene1_bin[k]

    gene_gene_interaction_name = []  # 这个需要根据最后总的互作基因名进行获取
    for i_gene_name in range(len(gene_name)):
        i_gene_name_2 = i_gene_name + 1  # 保证比i_gene_name大一个，防止有一样的互作名
        while i_gene_name_2 < len(gene_name):
            i_gene_name_3 = '%s,%s' % (gene_name[i_gene_name], gene_name[i_gene_name_2])
            gene_gene_interaction_name.append(i_gene_name_3)
            i_gene_name_2 = i_gene_name_2 + 1


    return  gene_location,genome_length,gene_direction,gene_locus_tag,gene_name,gene_distance,gene_length,gene1_bin,gene_n,gene1_bin_merge,gene_gene_interaction_name,genome_length_chr,gene_location_chr
