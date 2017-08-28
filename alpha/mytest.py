import ef_with_pandas

with open("/Users/xingyuwei/Downloads/human_rna_fna_refseq_mRNA_22389") as f:
    lines = f.readlines()

seq = lines[1][:-1]
print(ef_with_pandas.extract_features_using_dict(lines[0], seq))


