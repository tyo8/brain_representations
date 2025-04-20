import os
import re
import glob
import numpy as np
import pandas as pd

# global default argument values:
def_fig_size = (24, 24)
sample_dirnames = {"perm": "permtesting", "bstrap": "subsampling"}

############################################# DATA LOADING FUNCTIONS ###################################################
########################################################################################################################
# load_type (str) -- "solo", "pair", or "ext"
def _load(
        input_fpath, 
        load_type=None,
        args=None,
        data_only=True,
        enforce_match=False,
        permtype=None,
        check_pval=True, 
        extrema_only=False,
        debug=False
        ):
    df = pd.read_csv(input_fpath, index_col=0)

    if df.empty:
        print(f"pulled empty dataframe from path: \n{input_fpath}")
        return df
    else:
        df = _unify_df(df, fpath=input_fpath, enforce_match=enforce_match)


    if load_type == "pair":
        
        if check_pval:
            if "empirical_pval" in df.columns.values:
                df.drop( columns=["empirical_pval"], inplace=True )

        if data_only:
            null_mask = df["datatype"].str.contains("Null")
            if (permtype is not None) and ("permtype" in df.columns.values):
                null_mask = null_mask & (df["permtype"] == permtype)

            null_df = df[null_mask].copy()
            data_df = df[~null_mask].copy()

            data_df["Wp_XYNull_mean"] = np.mean(null_df["Wp_XY"])
            data_df["Wp_XYNull_std"] = np.std(null_df["Wp_XY"])
            data_df["permtype"] = permtype 

            df = data_df

    elif load_type == "ext":
        if "permtype" in df.keys():
            if args is not None:
                err_str = f"Specified permtype {args.permtype} not in df.permtype: \'{df.permtype.unique()}\'. From input path: \n{input_fpath}"
                assert args.permtype in df.permtype.unique(), err_str
                df = df[ df.permtype == args.permtype ]
            permtype = df["permtype"].unique()[0]
        else:
            permtype = "Empty"

        df["permtype"] = permtype

        if debug:
            if "datatype" not in df.keys():
                print("column 'datatype' not found in:", input_fpath)
                print("columns:", df.columns.values)
                print("dataframe", df)
                exit()

        null_mask = df["datatype"].str.contains("Null")
        null_df = df[null_mask].copy()
        data_df = df[~null_mask].copy()
        data_df["Wp_XYNull_min"] = np.min(null_df["Wp_XY"])
        data_df["Wp_XYNull_max"] = np.max(null_df["Wp_XY"])

        if not extrema_only:
            data_df["Wp_XYNull_mean"] = np.mean(null_df["Wp_XY"])
            data_df["Wp_XYNull_std"] = np.std(null_df["Wp_XY"])

        df = data_df
    elif load_type == "solo":
        df = df
    else:
        raise ValueError(f"Unrecognized 'load_type': {load_type}")

    return df
########################################################################################################################

########################################################################################################################
def _unify_df(df, fpath=None, enforce_match=False):
    if df.empty:
        return df

    # I'm never changing a naming convention again, even if it's terrible; this has been so annoying to deal with
    cols = [ col for col in df.columns.values if ("metric" in col) or ("name" in col) ]
    df[cols] = df[cols].applymap( lambda x: x.replace("Psim_ztrans", "Psim-ztrans") )

    if "X_type" in df.columns.values:
        df.rename( mapper={"X_type":"X_name"}, axis=1, inplace=True )
        if "Y_type" in df.columns.values:
            df.rename( mapper={"Y_type":"Y_name"}, axis=1, inplace=True )
            df[["X_modality","X_feature","X_metric"]] = df["X_name"].str.split('_', n=2, expand=True)
            df[["Y_modality","Y_feature","Y_metric"]] = df["Y_name"].str.split('_', n=2, expand=True)

    if not "Y_name" in df.columns.values:
        if "X_name" in df.columns.values:
            df[["modality","feature","metric"]] = df["X_name"].str.split('_', n=2, expand=True)
        else:
            df["X_name"] = df.apply( lambda x: "_".join([x["modality"], x["feature"], x["metric"]]), axis=1 )

        if fpath is not None and enforce_match:
            modality, feature, metric = _parse_fpath(fpath, metric=True)
            pars = {
                    "modality": modality,
                    "feature": feature,
                    "metric": metric
                    }
            for keyname in ["modality", "feature", "metric"]:
                if keyname not in df.columns.values:
                    df[keyname] = pars[keyname]
                else:
                    assert all(df[keyname] == pars[keyname]), f"Data value and filename conflict for key \'{keyname}\' in file: \n\'{fpath}\'"

        df[["modality","rank"]] = df.apply( lambda x: _pull_rank(x["modality"]), result_type="expand", axis=1 )
        df["feat_num"] = df.apply( lambda x: _pull_feat_num(x["rank"], x["feature"]), axis=1 )
    else:
        df[["X_modality","X_feature","X_metric"]] = df["X_name"].str.split('_', n=2, expand=True)
        df[["Y_modality","Y_feature","Y_metric"]] = df["Y_name"].str.split('_', n=2, expand=True)


    if "permtype" in df.columns.values:
        try:
            null_mask = df["datatype"].str.contains("Null")
            df.loc[null_mask, "datatype"] = df.loc[null_mask,:].apply( lambda x: "_".join([str(x["permtype"]), str(x["datatype"])]), axis=1 )
        except TypeError:
            print(f"encountered unexpected float values in \'permtype\' field: {list(set(df.permtype))}")
            print("attempting to resolve by forcing type to \'str\'.")

    if "permlabel" in df.columns.values:
        df.drop( labels = ["permlabel"], axis=1, inplace=True )

    if "taglist" in df.columns.values:
        df.drop( labels = ["taglist", "X_path"], axis=1, inplace=True )
        if "Wp_XY" not in df.columns.values:
            renamer={"Wp_XXhat_i":"Wp_XY", "PDXhat_diag_i":"PDY_diag"}
            df = _add_data_slice(df)
        else:
            renamer={"Wp_XhatYhat_i":"Wp_XY"}
            to_drop = ["Wp_XY", "Y_path", "Wp_YYhat_i", "Wp_XXhat_i", "dIM_YYhat_i", "PDXhat_diag_i", "PDYhat_diag_i",
                    "dIM_XXhat_i", "Mean Wp Approximation Difference", "Wphat_XY", "Wphat0_XY_i"]
            df.drop( labels = to_drop, axis=1, inplace=True )
        df.rename( mapper=renamer, axis=1, inplace=True )

    return df

def _add_data_slice(df):
    df.loc[-1] = df.loc[0].copy()
    df.loc[-1, "datatype"] = "Data"
    df.loc[-1, "Wp_XXhat_i"] = 0
    df.loc[-1, "PDXhat_diag_i"] = df.loc[-1, "PDX_diag"]
    return df
########################################################################################################################
# Data-wrangling helper functions
########################################################################################################################
def _get_fpath_list(args, debug=False):
    if args.pattern_restriction is None:
        args.pattern_restriction = ""

    basedir_pattern = os.path.join(
            args.input_dir,
            args.dir_pattern, 
            sample_dirnames[args.sample_type]
            )

    if args.sample_type=="perm":
        path_ext = os.path.join(f"X_*{args.pattern_restriction}*_dists", f"{args.f_pattern}.csv")
    elif args.sample_type == "bstrap":
        path_ext = f"bsdists_*{args.pattern_restriction}*.csv"

    match_pattern = os.path.join(
            basedir_pattern,
            path_ext
            )

    fpath_list = glob.glob(match_pattern)
    fpath_list.sort()

    if args.verbose:
        print(f"general match pattern is: \n\'{match_pattern}\'")

    if args.enforce_match and args.verbose:
        print("enforcing modality, feature, and metric matching between data and null")
        if debug:
            print(f"fpath_list has {len(fpath_list)} entries prior to match enforcement.")
        fpath_list = [fpath for fpath in fpath_list if '_'.join(_parse_fpath(fpath, metric=False)) in os.path.basename(fpath)]
        if debug or args.verbose:
            print(f"fpath_list has {len(fpath_list)} entries after match enforcement.")
    return fpath_list

def _pull_rank(long_method, debug=False):
    if 'PROFUMO' in long_method:
        rank=33
        method="PROFUMO"
    elif 'Glasser' in long_method:
        rank=360
        method="Glasser"
    else:
        rank_pattern = re.compile('\d{1,4}')
        rank = re.search(r'\d{1,4}', long_method).group()
        method = long_method.replace(rank,'')
        if debug:
            print(f"[method, rank] = {[method, int(rank)]}")
    return method, int(rank)

def _pull_feat_num(rank, feature):
    if isinstance(rank, float):
        rank = int(10**rank)    # assumes that non-integer 'rank' is actually log10(rank)

    if 'NM' in feature:
        feat_num = rank * (rank - 1) / 2
    elif 'Map' in feature:
        feat_num = rank * 91282
    elif 'Amps' in feature:
        feat_num = rank
    else:
        raise Exception("Unrecognized feature type")
    return int(feat_num)

def _parse_fpath(fpath, metric=True):
    if "subsampling" in fpath:
        longname = os.path.basename(fpath).split('.')[0].replace("bsdists_","")
    else:
        longname = os.path.basename(os.path.dirname(fpath))
    name = longname.replace("_dists","").replace("X_","")
    modality, feature, metric = name.split('_', maxsplit=2)
    if metric:
        return modality, feature, metric
    else:
        return modality, feature

def _get_fpath_types(fpath, dist_type="single"):
    if dist_type == "single":
        dirname = os.path.dirname(fpath)
        if "permtesting" in fpath:
            name = '_'.join(_parse_fpath(fpath))
            basedir = os.path.dirname(os.path.dirname(os.path.dirname(fpath)))
        elif "subsampling" in fpath:
            name = os.path.basename(fpath).replace('.csv','').replace('bsdists_','')
            basedir = os.path.dirname(os.path.dirname(fpath))
        else:
            raise ValueError(f"did not see expected filepath sampling convention name for distance type \'{dist_type}\' in given fpath: \n{fpath}")

        featnullspath = os.path.join(basedir, "permtesting", f"X_{name}_dists", f"data_vs_featurenull_{name}.csv")
        subjnullspath = os.path.join(basedir, "permtesting", f"X_{name}_dists", f"data_vs_subjectnull_{name}.csv")
        subsamplepath = os.path.join(basedir, "subsampling", f"bsdists_{name}.csv")
        outdir = basedir
    elif dist_type == "pair":
        xname = '_'.join(_parse_fpath(fpath))
        yname = re.split(r'vs.', os.path.basename(fpath))[1].split('_null')[0]
        basedir = os.path.dirname(os.path.dirname(os.path.dirname(fpath)))
        name = f"{xname}_vs_{yname}"

        featnullspath = os.path.join(basedir, "All_vs_AllNull", f"X_{xname}_dists", f"{xname}_vs_{yname}_null-featurePerms.csv")
        subjnullspath = os.path.join(basedir, "All_vs_AllNull", f"X_{xname}_dists", f"{xname}_vs_{yname}_null-subjectPerms.csv")
        subsamplepath = os.path.join(basedir, "All_vs_self", f"X_{xname}_dists", f"bspairdists_{xname}-vs-{yname}.csv")
        outdir = os.path.join(basedir, "single-pair_figures")
    else:
        raise ValueError(f"Unrecognized distance output file type \'{dist_type}\'")

    return (featnullspath, subjnullspath, subsamplepath), outdir, name


def merged_dfs(fpath_list, dist_type="single", debug=False, verbose=False):
    fpath_groups = list(set([ _get_fpath_types(fpath, dist_type=dist_type)[0] for fpath in fpath_list ]))
    merged_df_list = [ get_merge_df(group) for group in fpath_groups ]

    if verbose:
        if dist_type=="single":
            print(f"evaluating {len(merged_df_list)} merged dataframes.")
            for i,df in enumerate(merged_df_list):
                name = os.path.basename(fpath_groups[i][-1]).replace("bsdists_","").replace(".csv","")
                if debug:
                    datamask = (df.datatype=="Subsamp") | (df.datatype=="Data")
                    df_data = df.loc[datamask,:]
                    if len(df_data) < 1000:
                        print(f"found incomplete or empty subsampling/data (shape={df_data.shape}) at path: \n{fpath_groups[i][-1]}")
                        _ = _get_orig_bars(fpath_groups[i][-1], homdim=1)
                    else:
                        print(f"subsampling data (shape={df_data.shape}) passes inspection -- found at path: \n{fpath_groups[i][-1]}")
                print(f"merged_df from \'{name}\' paths has shape {df.shape}\n")

    if debug:
        ### debugging code ###
        merged_df_list[-1].to_csv("df_tmp.csv")       
        with open("fpath_group.txt",'w') as fin:
            for fpath in fpath_groups[-1]:
                fin.write(f"{fpath}\n")
        ### debugging code ###
    return merged_df_list

def get_merge_df(fpath_group):
    df_list = [pd.read_csv(fpath, index_col=0) for fpath in fpath_group if os.path.isfile(fpath)]

    for df in df_list:
        if df.empty:
            continue
        else:
            df = _unify_df(df)

    if all([df.empty for df in df_list]):
        print("\nCompletely empty dataframe output corresponding to fpath group:")
        for fpath in fpath_group:
            print(f"\t{fpath}")
        merge_df = pd.DataFrame(None)
    else:
        merge_df = pd.concat( df_list, ignore_index=True )

    return merge_df

def get_better_names(dist_type):
    if dist_type=="single":
        better_Wp_name = "Wasserstein distance (full vs. null/subsamp)"
    elif dist_type=="pair":
        better_Wp_name = "Wasserstein distance (paired full/null/subsamp BRs)"

    renamer = {
            "Wp_XY": better_Wp_name, 
            "PDY_diag": "Wasserstein distance from empty diagram",
            "modality": "Brain parcellation", 
            "feature": "Feature",
            "metric": "Metric"
            }
    denamer = {v: k for k, v in renamer.items()}      # inverse dictionary of 'renamer'
    return renamer, denamer


def _write_list(outpath, list_out):
    with open(outpath, 'w') as fout:
        fout.write(list_out.__str__())

def _write_img(fig, outpath, fig_size=def_fig_size):
    if fig_size is not None:
        fig.set_size_inches(fig_size, forward=False)
    if not os.path.isdir(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))
        Warning(f"Created new output directory: \n{os.path.basename(outpath)}")
    fig.savefig(outpath, dpi=600)
    print(f"saved to {outpath}")
########################################################################################################################



########################################################################################################################
########################################################################################################################


############################################### DEBUGGING FUNCTIONS ####################################################
########################################################################################################################
def _get_orig_bars(
        bsdist_fpath, 
        homdim=1, 
        basedir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/full-scale-expmt"
        ):
    xname = os.path.basename(bsdist_fpath).replace("bsdists","phom_data").replace(".csv","_dists")
    searchdir = os.path.join(basedir, "within*", "*_*", xname)
    
    [bars_fpath] = glob.glob(os.path.join(searchdir, "phom_X.txt"))     # should have exactly 1 result
    barlines = []
    with open(bars_fpath, 'r') as fin:
        phom = fin.read().split('\n')

    idx = phom.index("persistent homology intervals in dim 1:")
    barlines = phom[idx+1:-1]           # assumes all files have \n or EOF char as last line
    print(f"Found {int(len(barlines)/3)} H1 bars in original data at path: \n{bars_fpath}:")
    if len(barlines) > 0:
        print("with values:")
        for line in barlines:
            print(line)
        if len(glob.glob(os.path.join(searchdir, "B1match_dim1_n1000.txt"))) > 0:     # should have <=1 result
            [b1match_fpath] = glob.glob(os.path.join(searchdir, "B1match_dim1_n1000.txt"))     # should have <=1 result
            b1match_vec = np.loadtxt(b1match_fpath)
            print(f"Of {len(b1match_vec)} recorded attempted matches, {np.count_nonzero(b1match_vec)} produced matches with nonzero affinity. ref:\n{b1match_fpath}")
        else:
            print("-----No corresponding B1match file found!-----")
    return None
########################################################################################################################

