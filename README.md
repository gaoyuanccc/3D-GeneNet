# 3D-GeneNet
Constructing gene association networks using chromosomal conformational capture technology based on three-dimensional space (3D-GeneNet)

E-mail: 18434753515@163.com
## Introduction

3D-GeneNet is a tool for constructing gene association networks that use the chromosomal conformational capture data to estimate the spatial interaction frequency between genes and identify chromosomal interaction regions (CIDs). With normalized or unnormalized IF matrices and bacterial genome-wide Genbank annotation files as inputs, the software will construct the gene/protein interaction network, identify the CIDs, and export the network as a file.


## Installation
Requirement:
* python (3.x)
* biopython
* numpy
* pandas
* scipy

Notes:
You may alternatively install using the following `conda` commands shown below, which are derived from our built environment file `environment.yaml`.

    conda env create -f environment.yaml -n new_env_name


##  How to use 3D-GeneNet?

### Parameters
| Parameter |Full name| Description| Default 
|:-:|:-:|--|:-:|
|-gb|genbank_file| Genbank filename with annotation information for the strain. |\|
|-p|multichromosome|Used to determine if the target strain is multichromosome; if the strain is multichromosome, set to 1 and is 0 by default.|0|
|-gbc|multichromosomeGenbank|If the strain is multichromosome, enter the Genbank filenames for the chromosomes in order, separated by a '/' symbol.|\|
|-if|interaction_file| Contact matrix filename to convert interaction frequencies between bins to interaction frequencies between genes. |\|
|-r|resolution|Resolution of interaction_file.|\|
|-i|input_file_path|Path to the input file directory.|\|
|-b|bin_begin_number |Begin the bin serial number in the contact matrix, which is 1 by default.|1|
|-n|n_bootstrap|Number of bootstrap iterations,which is 1000 by default.|1000|
|-s|seed|Seed for random number generation to ensure reproducibility.|None|
|-q|quantile_threshold|This parameter selects the percentile of p-values, with the default value set at 95. This means it filters for the top 5% of gene pairs based on their p-values.|95|
|-d|remove_distance|Linear genomic distances to select removed gene pairs and is 0 by default.|0|
|-ic|cid_interaction_file|Contact matrix filename to identify CIDs and is interaction_file by default. If cid_interaction_file = None, then it will not identify CIDs|\|
|-rc|resolution_cid|Resolution of CIDmatrix.|\|
|-o|outfile_path|Output folder name.|output|
|-h|help|Help documentation|\|

### Usage

    python 3D-GeneNet.py -i [input_file_path] -gb [genbank_file] -if [interaction_file] -ir [resolution] [...]

#### For example

 - Single-chromosome
The GSM2870409 sample of *Escherichia coli* K-12 MG1655 was taken is an example to demonstrate the complete function.
````
# Run 3D-GeneNet
python 3D-GeneNet.py -i "D:\Data\Project\3D-GeneNet\software_CLI\3D-GeneNet\input\ecoli" -gb U00096.3.gb -if rawdata_1000_iced.matrix -ic rawdata_10000_iced.matrix -o "D:\Data\Project\3D-GeneNet\software_CLI\3D-GeneNet\output\ecoli" -r 1000 -rc 10000 -n 1000 -q 99.5
------------------------------
2023-12-05 22:13:48
Run 3D-GeneNet

------------------------------
2023-12-05 22:13:48
Convert interaction frequency

------------------------------
2023-12-05 22:14:49
Select gene pairs

------------------------------
2023-12-05 22:15:10
Identify CIDs

------------------------------
2023-12-05 22:15:14
Output files

Time:    89.31265020370483s
````

 - Multi-chromosome
 The SRR3180951 sample of *Vibrio cholerae*  was taken is an example to demonstrate the complete function.
 ````
 #Run 3D-GeneNet
python 3D-GeneNet.py -i "D:\Data\Project\3D-GeneNet\software_CLI\3D-GeneNet\input\vibrio_cholerae" -p 1 -gbc vibrio_cholerae_1.gb/vibrio_cholerae_2.gb -if SRR3180951_1000_iced.matrix -ic SRR3180951_10000_iced.matrix -o "D:\Data\Project\3D-GeneNet\software_CLI\3D-GeneNet\output\vibrio_cholerae" -r 1000 -rc 10000 -n 1000 -q 99.5
------------------------------
2023-12-05 22:24:30
Run 3D-GeneNet

------------------------------
2023-12-05 22:24:30
Convert interaction frequency

------------------------------
2023-12-05 22:25:01
Select gene pairs

------------------------------
2023-12-05 22:25:12
Identify CIDs

------------------------------
2023-12-05 22:25:15
Output files

Time:    46.110193729400635s
 ````

#### Detailed usage

 1. -i input_file_path
Put all input files into the  'input_file_path', folder before starting the run. 'input' is the folder name where all input files are located.
 2. -o outfile_path
 'outfile_path' is the name of the folder where all output files are located. The software will output 5 files if it identifies CIDs. '*cid_t_pvalue.csv*' and '*cid_bin_preference.pdf*' are files describing bins' t and pvalue. *’cid_number.csv*' is the range of CIDs. '*gene_association_network.txt*' and '*gene_cid.txt*' are files from the obtained gene association network and the CID attribute of the gene. '*gene_association_network.txt*' is visualized by Cytoscape. Its specific structure is as follows:
 ```
	+ input
		++ U00096.3.gb (genbank_file)
		++ GSM2870409_1000_iced.matrix (interaction_file)
		++ GSM2870409_10000_iced.matrix (cid_interaction_file)
	+ output
		++ cid_bin_preference.pdf
		++ cid_number.csv
		++ cid_t_pvalue.csv
		++ gene_cid.csv
		++ gene_association_network.csv
```
 3.  -gb genbank_file
 This command inputs the Genbank annotation file of the strain, which needs to contain the genome length and the position information of the gene, and its format is as follows:
 ````
 LOCUS       U00096               4641652 bp    DNA     circular BCT 23-SEP-2020
DEFINITION  Escherichia coli str. K-12 substr. MG1655, complete genome.
            ......
            ......
     gene            190..255
                     /gene="thrL"
                     /locus_tag="b0001"
                     /gene_synonym="ECK0001"
                     /db_xref="ASAP:ABE-0000006"
                     /db_xref="ECOCYC:EG11277"
     CDS             190..255
                     /gene="thrL"
                     /locus_tag="b0001"
                     /gene_synonym="ECK0001"
                     /codon_start=1
                     /transl_table=11
                     /product="thr operon leader peptide"
                     /protein_id="AAC73112.1"
                     /db_xref="UniProtKB/Swiss-Prot:P0AD86"
                     /db_xref="ASAP:ABE-0000006"
                     /db_xref="ECOCYC:EG11277"
                     /translation="MKRISTTITTTITITTGNGAG"
````
 4.  -im interaction_file -r resolution
This command inputs the contact matrix file that converts the interaction frequency is three column data. The resolution is this matrix resolution, 1000 is recommended.	
	**Three columns format**
	The data consists of three columns: the first column is the serial number of the first bin, the second column is the serial number of the second bin, and the third column is the interaction frequency of the two bins.
        ``````
        0	0	65.478946
        0	1	633.672990
        0	2	9.645624
        0	3	5.717007
        0	4	5.582881
        ...	...	...
        4641	4641	532.456055
        ``````
 5.  -ic --cid_interaction_file -rc resolution_cid
 If you need to identify CIDs, provide the contact matrix in the same format as the above interaction_file; 10000 is the suggested resolution. If you do not need to identify CIDs, enter the command `-ic None`.
 6. -p --multichromosome -gbc --multichromosomeGenbank
 If the strain is multi-chromosome, set -p 1. Genbank filenames for chromosomes are entered in order after -gbc, separated by a '/'.

 

 
 

 


