import os
import sys
import datetime
import lindecomp_stability as ld_stb


dataset_listname=sys.argv[1]
output_basedir=sys.argv[2]
listpath=sys.argv[3]
decomp_method=sys.argv[4]
if len(sys.argv) < 6:
    par = False
else:
    par = True
    n_workers = int(sys.argv[5])

def par_stbl_iters(output_bdir):
    brainreps_data = ld_stb.load_reps(dataset_listname)

    ld_stb.run_stability_iters(output_bdir, listpath=listpath, decomp_method=decomp_method,
            read_mode=False, reps=brainreps_data,write_mode=False)

def run_parallel_iters(output_basedir,n_workers):
    import multiprocessing as mp
    
    date_label = datetime.date.today().strftime("%Y%b%d")
    dirlocs = [os.path.join(output_basedir,decomp_method+"_stbl_" + str(k) +"_" + date_label) for k in range(n_workers)]
    
    with mp.Pool(n_workers) as P:
        P.map(par_stbl_iters,dirlocs)

    decomp_vals = ld_stb.aggregate_decomps(dirlocs)

    stbl_arch_dir = os.path.join(output_basedir,decomp_method+"_stbl_paragg_" + date_label)
    if not os.path.isdir(stbl_arch_dir):
        os.makedirs(stbl_arch_dir)
        print("WARNING: Created directory " + stbl_arch_dir)

    return decomp_vals,stbl_arch_dir


if par:
    decomp_vals,stbl_arch_dir = run_parallel_iters(output_basedir,n_workers)
    ld_stb.save_stability_decomps(decomp_vals,stbl_arch_dir,decomp_method)

    ld_stb.summarize_iters(decomp_vals, output_basedir, decomp_method=decomp_method)
else:
    ld_stb.stability_testing(output_basedir, dataset_list_name=dataset_listname, 
            listpath=listpath, decomp_method=decomp_method)
