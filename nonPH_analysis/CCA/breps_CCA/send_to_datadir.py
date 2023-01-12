import os
import sys
import csv
import dill


working_dir = sys.argv[1]
outdir = sys.argv[2]

fin_loclist = [os.path.abspath(i) for i in os.listdir(working_dir) if ".CCA_res" in i]
foutnames = [i.split('.')[0] + "_cancorr_data.csv" for i in os.listdir(working_dir) if '.CCA_res' in i]
fout_loclist = [os.path.join(outdir,i) for i in foutnames]

for i in range(len(fout_loclist)):
	with open(fin_loclist[i],'rb') as fin:
		data = dill.load(fin)
	with open(fout_loclist[i],'w') as fout:
		write = csv.writer(fout)
		write.writerow(data.cancorrs)
