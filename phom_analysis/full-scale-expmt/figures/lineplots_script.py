from distributional_summaries import summ_lineplots as SL
import pandas as pd
import os


stat_types = [i.split('_')[0] for i in os.listdir() if i.endswith('.csv')]
dim_types = ["feat_num", "rank"]
for dim_type in dim_types:
    stat_dfs = [pd.read_csv(i) for i in os.listdir() if i.endswith('.csv')]
    for i, stat_type in enumerate(stat_types):
        SL(stat_dfs[i], stat_type, dim_type=dim_type)
