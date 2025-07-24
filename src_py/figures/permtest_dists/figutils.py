import os
import re
import glob
import copy
import itertools
import functools
import numpy as np
import pandas as pd
import seaborn as sns
import figstats as fstats
from numbers import Number
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform

# global default argument values:
def_fig_size = (24, 24)
sample_dirnames = {"perm": "permtesting", "bstrap": "subsampling"}
order_fpath = "/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/src_py/figures/permtest_dists/xnames_order.csv"

############################################# DATA LOADING FUNCTIONS ###################################################
########################################################################################################################
# load_type (str) -- "solo", "pair", or "ext"
def _load(
        input_fpath, 
        load_type=None,
        enforce_match=False,
        permtype=None,
        check_pval=True, 
        pval_args=None,
        extrema_only=False,
        debug=False
        ):
    df = pd.read_csv(input_fpath, index_col=0)

    if debug:
        print(f"before _unify_df: \n{df}")

    if df.empty:
        print(f"pulled empty dataframe from path: \n{input_fpath}")
        return df
    else:
        df = _unify_df(df, fpath=input_fpath, enforce_match=enforce_match)

    if debug:
        print(f"after _unify_df: \n{df}")

    if load_type == "pair":

        if "permtype" not in df.keys():
            if permtype is None:
                print(f"No \'permtype\' found or given in {df.keys()}. Loaded from: \n{input_fpath}\n")
            else:
                df["permtype"] = permtype
        
        if check_pval:
            if "empirical_pval" in df.keys():
                df.drop( columns=["empirical_pval"], inplace=True )
            if pval_args is not None:
                fstats._add_emp_pval(
                    df, 
                    permtype=pval_args.permtype, 
                    tail_type=pval_args.tail_type, 
                    corr_type=pval_args.corr_type, 
                    null_hi=pval_args.null_hi, 
                    null_lo=pval_args.null_lo
                    )
            else:
                err_msg = f"No p-value type variable in {df.columns.values}. Read from: \n{input_fpath}\n"
                assert any(["pval" in varname for varname in df.columns.values]), err_msg

        null_mask = df["datatype"].str.contains("Null")
        if (permtype is not None) and ("permtype" in df.columns.values):
            null_mask = null_mask & (df["permtype"] == permtype)

        df.loc[~null_mask,"Wp_XYNull_mean"] = np.mean(df.loc[null_mask,"Wp_XY"])
        df.loc[~null_mask,"Wp_XYNull_std"] = np.std(df.loc[null_mask,"Wp_XY"])
        df.loc[~null_mask,"permtype"] = permtype 
        df = df[ ~null_mask ]

    elif load_type == "ext":
        if "permtype" in df.keys():
            if permtype is None:
                permtypes = df["permtype"].unique()
            else:
                err_str = f"Specified permtype {permtype} not in df.permtype: \'{df.permtype.unique()}\'. From input path: \n{input_fpath}"
                assert permtype in df.permtype.unique(), err_str
                permtypes = [permtype]
        else:
            assert len(df.datatype.unique())==1, f"column \'permtype\' not found in df.keys() even though multiple datatypes present: \n{df}"
            return pd.DataFrame([])

        if debug:
            if "datatype" not in df.keys():
                print("column 'datatype' not found in:", input_fpath)
                print("columns:", df.columns.values)
                print("dataframe", df)
                exit()

        allnull_mask = df["datatype"].str.contains("Null")

        df_data = pd.DataFrame(
                data = np.repeat(
                    df[ ~allnull_mask ].values, 
                    len(permtypes), 
                    axis=0),
                columns = df.columns)

        for i, permtype in enumerate(permtypes):
            perm_mask = df["permtype"] == permtype
            null_mask = allnull_mask & perm_mask

            df_data.loc[i, "permtype"] = permtype

            if df[null_mask].empty:
                if debug:
                    print(f"BAD loaded dataframe: \n{df}")
                    print(f"from file: \n{input_fpath}")
                    exit()
                return df[null_mask]
            else:
                if debug:
                    print(f"GOOD loaded dataframe: \n{df}")
                    print(f"from file: \n{input_fpath}")

                df_data.loc[i,"Wp_XYNull_min"] = np.min(df.loc[null_mask,"Wp_XY"])
                df_data.loc[i,"Wp_XYNull_max"] = np.max(df.loc[null_mask,"Wp_XY"])
                df_data.loc[i,"Wp_XYNull_mean"] = np.mean(df.loc[null_mask,"Wp_XY"])
                df_data.loc[i,"Wp_XYNull_std"] = np.std(df.loc[null_mask,"Wp_XY"])

        df = df_data
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
    df[cols] = df[cols].map( lambda x: x.replace("Psim_ztrans", "Psim-ztrans") )

    if "X_type" in df.columns.values:
        df.rename( mapper={"X_type":"X_name"}, axis=1, inplace=True )
        if "Y_type" in df.columns.values:
            df.rename( mapper={"Y_type":"Y_name"}, axis=1, inplace=True )
            df[["X_modality","X_feature","X_metric"]] = df["X_name"].str.split('_', n=2, expand=True)
            df[["Y_modality","Y_feature","Y_metric"]] = df["Y_name"].str.split('_', n=2, expand=True)

    if "Y_name" in df.columns.values:
        df[["X_modality","X_feature","X_metric"]] = df["X_name"].str.split('_', n=2, expand=True)
        df[["Y_modality","Y_feature","Y_metric"]] = df["Y_name"].str.split('_', n=2, expand=True)
        df[["X_modality","X_rank"]] = df.apply( lambda x: _pull_rank(x["X_modality"]), result_type="expand", axis=1 )
        df[["Y_modality","Y_rank"]] = df.apply( lambda x: _pull_rank(x["Y_modality"]), result_type="expand", axis=1 )
        df["X_feat_num"] = df.apply( lambda x: _pull_feat_num(x["X_rank"], x["X_feature"]), axis=1 )
        df["Y_feat_num"] = df.apply( lambda x: _pull_feat_num(x["Y_rank"], x["Y_feature"]), axis=1 )
    else:
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

    if "permtype" in df.columns.values:
        try:
            null_mask = df["datatype"].str.contains("Null")
            df.loc[null_mask, "datatype"] = df.loc[null_mask,:].apply( lambda x: "_".join([str(x["permtype"]), str(x["datatype"])]), axis=1 )
        except TypeError:
            print(f"encountered unexpected float values in \'permtype\' field: {df.permtype.unique()}")
            print("attempting to resolve by forcing type to \'str\'.")

    if "permlabel" in df.columns.values:
        df.drop( labels = ["permlabel"], axis=1, inplace=True )

    if "empirical_pval" in df.columns.values:
        df.drop( labels = ["empirical_pval"], axis=1, inplace=True )

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

# specialized glob function to allow equivalence classes of search terms and ^ ('not') logical operator
########################################################################################################################
def logical_glob(
        search_pattern, 
        exclude=False, exclude_terms=None, 
        verbose=False, debug=False):

    if debug:
        print("search parameters at outset", f"\nsearch_pattern = {search_pattern}", f"\nexclude = {exclude}", f"\nexclude_terms = {exclude_terms}")

    if exclude and (exclude_terms is not None):
        all_search = search_pattern
        for term in exclude_terms:
            all_search = all_search.replace(term,'*')
        all_results = glob.glob(all_search.replace('^',''))
        search_results = [ res for res in all_results if not any([x in res for x in exclude_terms ]) ]
    else:
        search_results = glob.glob(search_pattern)

    if verbose:
        print(f"Found {len(search_results)} matches to pattern: \n{search_pattern}")
        if exclude:
            print(f"(excluding {len(all_results) - len(search_results)} search results with exclude_terms={exclude_terms})")

    return search_results

########################################################################################################################
# Data-wrangling helper functions
########################################################################################################################
########################################################################################################################
def _get_symmetrized_data(
        alldata_grid, 
        symmetrized_vars=["Wp_XY"], 
        enforce_symmetry=True, 
        check_pval=True,
        set_order=False,
        order = order_fpath,
        verbose=True,
        debug=False
        ):
    xnamelist = [i[0]["X_name"].unique()[0] for i in alldata_grid]
    ynamelist = [j["Y_name"].unique()[0] for j in alldata_grid[0]]

    symmetric_dict = {}
    if check_pval:
        pval_vars = [ varname for varname in alldata_grid[0][0].columns.values if "pval" in varname ]
        symmetrized_vars = symmetrized_vars + pval_vars


    for varname in symmetrized_vars:
        try:
            vals = np.squeeze(np.array([[j[varname].to_numpy() for j in i] for i in alldata_grid]))
            if "pval" in varname:
                vals = _sym_pvals(vals, varname=varname)
            else:
                if all([ isinstance(entry, Number) for entry in vals.flatten() ]):
                    vals = _enforce_symmetry(vals, fill_val = 0)
                else:
                    vals = _enforce_symmetry(vals, fill_val = np.nan)
            if verbose:
                print(f"variable {varname} has grid of values with shape: {vals.shape}")
        except ValueError:
            new_entry = [[j[varname].to_numpy() for j in i] for i in alldata_grid]
            if debug:
                ### debugging code ###
                shape_vec = [ [ (len(var), i.shape) for i in var ] for var in new_entry ]
                name_grid = np.array([[(j["X_name"].unique().astype(str), j["Y_name"].unique().astype(str)) for j in i] for i in alldata_grid])
                futils._write_list("debug/shape_vec.txt", shape_vec)
                futils._write_list("debug/name_grid.txt", name_grid)
                print(f"found data inhomogeneity in {varname} readin. Saved grid of shapes and pair names to (resp.) paths:")
                print("debug/shape_vec.txt")
                print("debug/name_grid.txt")
                exit()
                ### debugging code ###

        symmetric_dict[varname] = vals
        print(f"\'{varname}\' gridded.")

    if debug:
        ### debugging code ###
        print(f"Names of {len(xnamelist)} 'X' spaces: \n{xnamelist}")
        print(f"Names of {len(ynamelist)} 'Y' spaces: \n{ynamelist}")
    if verbose:
        print(f"Entries in list of grid values have the following shapes: \n{[(var, symmetric_dict[var].shape) for var in list(symmetric_dict.keys())]}")
        # print("First entry in symmetric_dict: ", np.array(symmetric_dict[symmetrized_vars[0]]))
        print("")
        ### debugging code ###

    if enforce_symmetry:
        print(f"enforced symmetry in variables: \n{symmetrized_vars}\n")
        ynamelist = [xnamelist[0], *ynamelist]
        try:
            assert xnamelist == ynamelist
        except AssertionError:
            if debug:
                print(f"namelists are unequal in forced symmetric case! xnamelist: {len(xnamelist)} entries, ynamelist: {len(ynamelist)} entries")
                # print(f"namelists are unequal in forced symmetric case! \nxnamelist: {len(xnamelist)} entries\nynamelist: {len(ynamelist)} entries")
            ynamelist = xnamelist

    if set_order:
        if debug:
            print(f"xnamelist: \n{xnamelist}")
            print(f"ynamelist: \n{xnamelist}")
            print(f"symmetric_dict: \n{symmetric_dict}")
        [xnamelist, ynamelist], symmetric_dict = _reorder_arrdict([xnamelist.copy(), ynamelist.copy()], symmetric_dict.copy(), order=order)

    return xnamelist, ynamelist, symmetric_dict

## symmetrization helper functions
########################################################################################################################
# Enforces symmetry under assumption 'gridlist' produced by a pairwise process skipping its first trivial pairing
def _enforce_symmetry(mtx, debug=False, fill_val=np.nan):
    assert len(mtx.shape)==2, "Only valid for matrix inputs"
    if mtx.shape[0] == mtx.shape[1]:
        sym_mtx = (mtx + mtx.T)/2
    else:
        assert (mtx.shape[0]-1)==mtx.shape[1], f"Input matrix assumed to have shape (n,n-1): instead, given matrix has shape {mtx.shape}"

        # takes values from upper diagonal
        sym_mtx = squareform(triu_vals(mtx, k=0))
        np.fill_diagonal(sym_mtx, fill_val)

        if isinstance(sym_mtx[0][1], float):
            assert np.allclose(sym_mtx, sym_mtx.T, equal_nan=True), f"Symmetrization failed: \"sym_mtx\" is \n{sym_mtx}"

    return sym_mtx

def _sym_pvals(pval_mtx, varname=None):
    if varname is None:
        fill_val = 0
    else:
        fill_val = int("right" in varname)

        sym_pval = _enforce_symmetry(pval_mtx, fill_val=0)
        if "fdr" in varname:
            sym_pval = squareform(fstats.correct_pvals(squareform(sym_pval), corr_type="fdr"))
        elif "fwe" in varname:
            sym_pval = squareform(fstats.correct_pvals(squareform(sym_pval), corr_type="fwe"))
        np.fill_diagonal( sym_pval, fill_val )
        return sym_pval

def triu_vals(A, k=1):
    n = min(A.shape)
    vals = A[np.triu_indices(n, k)]
    return vals
########################################################################################################################




# Data wrangling functions
# may want to change logic/implementation of searches to allow for some kind of "or" | syntax processing -- convert to regex?
def _get_fpath_set(args, dist_type="single", set_type="list", debug=False):
    if args.pattern_restriction is None:
        exclude = False
        exclude_terms = ''
    else:
        exclude = ('^' in args.pattern_restriction)
        if exclude:
            exclude_terms = args.pattern_restriction.split('^')[1:]
            print(f"excluding terms containing any of: \'{exclude_terms}\'")
        else:
            exclude_terms = ''

    if dist_type == "single":
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

        fpath_list = logical_glob(match_pattern, exclude=exclude, exclude_terms=exclude_terms)
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

        fpath_set = fpath_list

    elif dist_type=="pair":

        args.dir_pattern='X_*_dists'
        args.f_pattern = '*_vs_*.csv'

        if args.pattern_restriction is not None and args.permtype is not None:
            if not args.output_dir.endswith(args.pattern_restriction.replace('^', 'not-').replace('*', '-and-')):
                args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)
                args.output_dir = args.output_dir.replace('^', 'not-').replace('*', '-and-')

            args.dir_pattern = args.dir_pattern.replace('_dists', f'*{args.pattern_restriction}*_dists')
            args.f_pattern = args.f_pattern.replace('_vs_', f'{args.pattern_restriction}*_vs_*{args.pattern_restriction}')

        if args.permtype is not None:
            args.f_pattern = args.f_pattern.replace(".csv",f"{args.permtype}Perms.csv")

        if set_type=="list":
            if args.fpathlist_path is None: 
                args.search_pattern = os.path.join(args.input_dir, args.dir_pattern, args.f_pattern)
                fpath_list = logical_glob(args.search_pattern, exclude=exclude, exclude_terms=exclude_terms)
            else:
                with open(args.fpathlist_path, 'r') as fin:
                    fpath_list = fin.read().split('\n')

            fpath_set = fpath_list

        if set_type=="grid":

            if args.fpathlist_path is None:
                args.pdir_pattern = os.path.join( args.input_dir, args.dir_pattern )
                dpath_list = logical_glob(args.pdir_pattern, exclude=exclude, exclude_terms=exclude_terms); dpath_list.sort()
                fpath_grid = [ logical_glob(os.path.join(dpath, args.f_pattern), exclude=exclude, exclude_terms=exclude_terms) for dpath in dpath_list ]
            else:
                with open(args.fpathlist_path, 'r') as fin:
                    fpath_list = fin.read().split('\n')
                dpath_list = list(set(list([ os.path.dirname( fpath ) for fpath in fpath_list ]))); dpath_list.sort()
                fpath_grid = [ [fpath for fpath in logical_glob(os.path.join(dpath, '*' ), exclude=exclude, exclude_terms=exclude_terms) if fpath in fpath_list ] 
                              for dpath in dpath_list ]

            fpath_grid = [ pathlist for pathlist in fpath_grid if pathlist ]    # removes empty lists (corresponding to directories with no successful search hits)
            [pathlist.sort() for pathlist in fpath_grid]
            fpath_set = fpath_grid

    return fpath_set

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

def _parse_fpath(fpath, pathtype="solo", metric=True):
    if pathtype=="solo":
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
    elif pathtype=="pair":
        fname = os.path.basename(fpath)
        dname = os.path.basename( os.path.dirname( fpath ))
        Y_name = fname.split('.')[0].split('vs')[1][1:]         # removes file extension, takes the right half after "vs", and removes the first character
        Y_name = Y_name.split('_null-')[0]                      # removes null-type specifications from 'Y_name' if present in filename
        X_name = dname.replace("_dists","").replace("X_","")    # removes extra text from X_name
        return X_name, Y_name

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
                    nullmask = df.datatype.str.contains("Null")
                    df_data = df.loc[datamask,:]
                    df_null = df.loc[nullmask,:]
                    if len(df_data) < 1000:
                        print(f"found incomplete or empty subsampling/data (shape={df_data.shape}) at path: \n{fpath_groups[i][-1]}")
                        _ = _get_orig_bars(fpath_groups[i][-1], homdim=1)
                    else:
                        print(f"subsampling data (shape={df_data.shape}) passes inspection -- found at path: \n{fpath_groups[i][-1]}")
                    if len(df_null) < 2000:
                        print(f"found incomplete or empty null data (shape={df_null.shape}) at path: \n{fpath_groups[i][:2]}")
                        #_ = _get_orig_bars(fpath_groups[i][0], homdim=1)
                        #_ = _get_orig_bars(fpath_groups[i][1], homdim=1)
                    else:
                        print(f"permutation data (shape={df_null.shape}) passes inspection -- found at path: \n{fpath_groups[i][:2]}")
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

def get_aesthetic_names(dist_type):
    if dist_type=="single":
        aesthetic_Wp_name = "Wasserstein distance (full vs. null/subsamp)"
    elif dist_type=="pair":
        aesthetic_Wp_name = "Wasserstein distance (paired full/null/subsamp BRs)"

    renamer = {
            "Wp_XY": aesthetic_Wp_name, 
            "PDY_diag": "Wasserstein distance from empty diagram",
            "modality": "Reduction algorithm", 
            "feature": "Feature",
            "metric": "Metric"
            }
    denamer = {v: k for k, v in renamer.items()}      # inverse dictionary of 'renamer'
    return renamer, denamer

def _nice_feats(featname):
    featname.replace("spNM","spatial NetMat")
    featname.replace("pNM","partial NetMat")
    featname.replace("NM","full NetMat")
    featname.replace("Amps","Amplitudes")
    return featname


def _write_list(outpath, list_out):
    with open(outpath, 'w') as fout:
        fout.write(list_out.__str__())

def _write_img(fig, outpath, fig_size=def_fig_size):

    outpath = outpath.replace('^','not-').replace('*','-and-')      # replace search operators with logical text

    if fig_size is not None:
        fig.set_size_inches(fig_size, forward=False)
    if not os.path.isdir(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))
        Warning(f"Created new output directory: \n{os.path.basename(outpath)}")
    fig.savefig(outpath, dpi=600)
    print(f"saved to {outpath}")
########################################################################################################################



########################################################################################################################
def _get_auc_mask(args=None, auc_df=None, fpath_list=None, alpha=None, debug=False):
    if auc_df is None:
        solo_args = copy.deepcopy(args)
        solo_args.enforce_match = True
        solo_args.write_mode = False

        if "permtype" not in vars(solo_args).keys():
            solo_args.permtype = "subject"
        if fpath_list is None:
            solo_args.ROC_analysis = True
            solo_args.AUC_filter = False
            solo_args.distribution_plots = False
            solo_args.aggregate_plots = False
            solo_args.solo_plots = False
            solo_args.input_dir = os.path.dirname(args.input_dir)
            solo_args.search_pattern = None
            solo_args.dir_pattern = "within_*"
            solo_args.sample_type = "bstrap"

            if debug:
                ### debugging code ###
                solo_args.verbose = True
                print(f"arguments initialized as: \n{solo_args}")
                ### debugging code ###
            else:
                solo_args.verbose = False

            from single_null_dists import main
            df = main(solo_args)
        else:
            from single_null_dists import make_AUC_plots
            solo_args.outdir = None
            df = make_AUC_plots(fpath_list, solo_args)

        if debug:
            ### debugging code ###
            print(f"auc_dataframe has values: \n{df}")
            ### debugging code ###
        
        df = df[ df["permtype"]==solo_args.permtype ]
    else:
        df = auc_df

    if args is None:
        err_msg = "must either specify \'alpha\' invidually or as part of argument suite."
        assert alpha is not None, err_msg
    else:
        alpha = args.alpha

    masks = [] 
    for var in df["ROC_variable"].unique():
        submask = df["ROC_variable"] == var
        subdf = df.loc[submask].set_index( "X_name" )        # assumes that specifying 'permtype' and 'ROC_variable' values give unique X_name
        masks.append( subdf["overlap"] < alpha )

    series_mask = functools.reduce(lambda x,y: x & y, masks)

    return series_mask


def _apply_series_mask(series_mask, xnamelist, ynamelist, symmetric_dict, debug=False):
    xbool = [ series_mask[xname] for xname in xnamelist ]
    ybool = [ series_mask[yname] for yname in ynamelist ]

    if debug:
        ### debugging code ###
        print(f"prior to masking:")
        print(f"\t(|xnamelist|, |ynamelist|) = {(len(xnamelist), len(ynamelist))}")
        print(f"\t(|xbool|, |ybool|) = {(len(xbool), len(ybool))}")
        print(f"\tsymmetric_dict has shapes: {[symmetric_dict[i].shape for i in symmetric_dict.keys()]}")
        ### debugging code ###

    # first, we check if symmetric_dict is a dictionary of matrices
    if isinstance(symmetric_dict, dict):
        for key in symmetric_dict.keys():
            try:
                values = symmetric_dict[key]
                xdrop = values[xbool,:]
                symmetric_dict[key] = xdrop[:, ybool]
            except IndexError as err:
                print(f"Failed with err: \n{err}")
                np.savetxt("values.txt",values)
                np.savetxt("xbool.txt", xbool)
                np.savetxt("ybool.txt", ybool)
                print(f"saved out offending data in \n{os.getcwd()}\nExiting.")
                exit()

    # otherwise, we check if symmetric_dict is nested list of lists:
    elif isinstance(symmetric_dict, list):
        if isinstance(symmetric_dict[0], list):
            symmetric_dict = list(itertools.compress(symmetric_dict, xbool))
            symmetric_dict = [ list(itertools.compress(sublist, ybool)) for sublist in symmetric_dict ]

    xnamelist = list(itertools.compress(xnamelist, xbool))
    ynamelist = list(itertools.compress(ynamelist, xbool))

    if debug:
        ### debugging code ###
        print(f"after masking:")
        print(f"\tlen(xnamelist, ynamelist) = {(len(xnamelist), len(ynamelist))}")
        print(f"\tsymmetric_dict has shapes: {[(i,symmetric_dict[i].shape) for i in symmetric_dict.keys()]}")
        ### debugging code ###

    return xnamelist, ynamelist, symmetric_dict


def get_pval_masks(symmetric_dict, alpha=None):
    dispvars = list(symmetric_dict.keys())
    if alpha is not None:
        # retains only display grid values corresponding to significant p-values
        pval_vars = [var for var in dispvars if "pval" in var]

        # create logical significance arrays
        sig_list = [ symmetric_dict[var] < alpha for var in pval_vars ]
        # 'mask' (in sns.heatmap) hides values at coordinate if mask(coord)=True; retain values by setting mask(coord)=False.
        masks = [ ~sig for sig in sig_list ]
        # masks = sig_list
        # show one unmasked plot as well (write it out last)
        pval_vars.append(None)
        masks.append(None)
    else:
        pval_vars = []
        masks = []

    for pval_var in pval_vars:
        if pval_var is not None:
            if 'two-tail' in pval_var:
                mask_var = pval_var + f"_INVmask{alpha}".replace("0.","")
                symmetric_dict[mask_var] = ~ masks[pval_vars.index(pval_var)]
            else:
                mask_var = pval_var + f"_mask{alpha}".replace("0.","")
                symmetric_dict[mask_var] = masks[pval_vars.index(pval_var)]

    return symmetric_dict

########################################################################################################################
def _reorder_arrdict(namelists, arrdict, order=order_fpath, verbose=True):
    keylist, arrlist = map(list, list(zip(*list(arrdict.items()))))

    if verbose:
        print(f"Reordering dictionary. Input dict has keys: \n{keylist}")

    namelists_ord, arrlist_ord = _reorder_arrays( namelists.copy(), arrlist.copy(), order=order, verbose=verbose )

    ord_dict = { keylist[i]: arrlist_ord[i] for i in range(len(keylist)) }

    return namelists_ord, ord_dict

# variable "order" is either a list, a filepath, or None
def _reorder_arrays(namelists, array_list, order=order_fpath, verbose=True):
    if isinstance(order,str):
        order_df = pd.read_csv(order, header=None)
        order_df.rename(columns = {0:"name"}, inplace=True)
    elif isinstance(order,list):
        order_df = pd.DataFrame(colunns = ["name"], data=order)
    elif order is None:
        order_df = pd.DataFrame(colunns = ["name"], data=sorted(xlist))

    err_msg = f"Found {len(namelists)} namelists but arrays have shapes {[len(arr.shape) for arr in array_list]}; match failed"
    assert len(namelists)==np.unique([len(arr.shape) for arr in array_list])[0], err_msg

    if verbose:
        print(f"re-ordering labels and arrays from ordering: \n{order}")
        print(f"original order: \n{namelists[0]}")

    newidx_list = [_reindex_list(namelist, order_df) for namelist in namelists]

    rename_idx_list = zip(namelists, newidx_list)

    namelists = [_reorder_list(ord_pair[0], ord_pair[1]) for ord_pair in rename_idx_list]
    array_list = [_reorder_array(arr, newidx_list) for arr in array_list]

    if verbose:
        print(f"new order: \n{namelists[0]}")
    return namelists, array_list

def _reindex_list(namelist, order_df, debug=True):
    if debug:
        print("namelist:", namelist)
        print(order_df)

    list_df = order_df[ order_df.name.apply( lambda x: x in namelist ) ]
    list_df.reset_index(drop=True, inplace=True)
    new_idx = [ list_df[ list_df.name == name ].index.values[0] for name in namelist ]

    if debug:
        print("new index:", new_idx)
        print("new name list:", [ namelist[i] for i in new_idx)
    return new_idx

def _reorder_list(inlist, new_idx, debug=False):
    ordlist = [None]*len(inlist)
    if debug:
        print(inlist)
        print(new_idx)
    for count, reidx in enumerate(new_idx):
        ordlist[reidx] = inlist[count]

    return ordlist

def _reorder_array(arr, idx_lists, debug=True):

    if arr.dtype == 'O':
        print(f"force-casting object-type array to string.")
        arr = arr.astype(str)

    if debug:
        print(f"reordering array: \n{arr}")

    ndims = len(arr.shape)
    for n in range(ndims):
        arr = np.rollaxis(arr, n)
        for count, reidx in enumerate(idx_lists[n]):
            arr[reidx,:] = arr[count,:]

    arr = np.rollaxis(arr,0)

    if isinstance(arr.dtype, str):
        print(f"filling diagnoal of force-casted array with NaNs.")
        np.fill_diagonal(arr, np.nan)

    if debug:
        print(f"reordered array: \n{arr}")
        print(f"with datatype: {arr.dtype}, or '{str(arr.dtype)}'")

    return arr
########################################################################################################################



########################################################################################################################

############################################### DEBUGGING FUNCTIONS ####################################################
## Debug bar-read/-extraction functions by pulling corresponding persistent homology raw output
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

## Debug dictionary of symmetric values 'symmetric_dict'
########################################################################################################################
def _debug_symmetric_dict(symmetric_dict):
    # del symmetric_dict['datatype']
    varlist = list(symmetric_dict.keys())
    print(f"data loaded into 'symmetric_dict' has (upper-triangular) shapes: \n{[(var, symmetric_dict[var].shape) for var in varlist]}")
    # print(f"data loaded into 'symmetric_dict' has first values: \n{[(var, symmetric_dict[var][0]) for var in varlist]}")
    outdir = "symmetric_dict"
    with open('symmetric_dict/symmetric_dict.npy','wb') as fout:
        np.save(fout, symmetric_dict)
    for var in varlist:
        fpath = os.path.join(outdir, f"{var}.txt")
        val = triu_vals(symmetric_dict[var])
        np.savetxt(fpath, val)
        print(f'{var} written to file:', os.path.join(os.getcwd(), fpath))
        symmetric_dict[var] = val
########################################################################################################################
########################################################################################################################


def proxy_legend(labels, palette=None):

    colorset = sns.color_palette(
            palette=palette, 
            n_colors = len(labels),
            as_cmap=False
            )
    handles = [plt.Rectangle((0,0), 1, 1, color=colorset[i]) for i,label in enumerate(labels)]
    return handles, labels
