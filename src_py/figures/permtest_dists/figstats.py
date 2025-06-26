import os
import ast
import copy
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import figutils as futils
from scipy import stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

# computes an empirical ROC curve (and AUC) given two data distributions.
########################################################################################################################
def do_ROC_analysis(
        df_list, 
        outdir='.', 
        distvars=["Wp_XY", "PDY_diag"], 
        dist_type="single",
        write_mode=True,
        verbose=False,
        debug=False
        ):

    nulls = ["subject", "feature"]

    auc_list = []
    for distvar in distvars:
        for null in nulls:
            auc_sublist = []
            for df in df_list:
                df_data = df.loc[df.datatype == "Subsamp", :]
                df_null = df.loc[df.datatype == f"{null}_Null", :]
                names = df["X_name"].unique()
                if verbose:
                    print(f"In {names}, df_data has shape {df_data.shape}, df_null has shape {df_null.shape}")
                if df_null.empty:
                    # Will hold if (only if) there was some error in computing null distances/this null set was not computed
                    if debug:
                        ### debugging code ###
                        print(f"empty df_null for nulltype=\'{null}\' pulled from: \n{df}\n")
                        print(f"df has datatypes={df.datatype.unique()} and permtypes={df.permtype.unique()}")
                        exit()
                        ### debugging code ###
                    auc = None
                    overlap = None
                elif df_data.empty:
                    # Will hold if H1 trivial for full data (nothing to bootstrap)
                    auc = None
                    overlap = None
                elif df["PDX_diag"].unique() < 1e-9:
                    auc = 0
                    overlap = 1
                else:
                    try:
                        assert len(names) == 1, "Conflating distributions for more than one brain representation type."
                    except AssertionError as err:
                        print(f"More than one name found: {names}")
                        print(f"Offending dataframe written to: \n{os.getcwd()}/debug/df_err.csv")
                        df.to_csv('debug/df_err.csv')
                        exit()

                    if dist_type == "single":
                        nullnames = df_null["X_name"].unique()
                        try:
                            assert names == nullnames, f"Comparing unmatched data and null distance distributions: \ndata={names} \nnull={nullnames}"
                        except ValueError as err:
                            print(f"failed for {names} with error: \n{err}\n")
                            df_data.to_csv('debug/df_data_err.csv')
                            df_null.to_csv('debug/df_null_err.csv')
                            print(f"offending dataframes saved to \'df_data_err.csv, df_null_err.csv\' in \'{os.getcwd()}/debug\'")
                            exit()


                    datavals = df_data[distvar].to_numpy()
                    nullvals = df_null[distvar].to_numpy()

                    # only counts as significant if subsamples are *closer* to original than null is
                    if distvar == "Wp_XY":
                        _, auc = get_roc(datavals, nullvals, flip = True)
                    elif distvar == "PDY_diag":
                        _, auc_l = get_roc(datavals, nullvals, flip = True)
                        _, auc_r = get_roc(datavals, nullvals)
                        auc = max(auc_l, auc_r)

                    overlap = (1 - auc)

                auc_dict = {
                        "X_name": str(list(names)[0]),
                        "permtype": null,
                        "ROC_variable": distvar,
                        "AUC": auc,
                        "overlap": overlap
                        }
                auc_list.append( auc_dict )
                auc_sublist.append( auc_dict )
            auc_subdf = pd.DataFrame( data=auc_sublist )
            if write_mode:
                outname = f"{distvar}-{dist_type}-distplot_AUC_summary_{null}-nulls.csv"
                outpath = os.path.join(outdir, outname)
                auc_subdf.to_csv(outpath)
                print(f"saved to {outpath}")

    if verbose:
        ### debugging code ###
        print(f"auc_list has length {len(auc_list)}")
        if debug:
            print(f"and values \n{auc_list}")
        ### debugging code ###
    auc_df = pd.DataFrame( data=auc_list )

    if write_mode:
        outname = f"{'-'.join(distvars)}-{dist_type}-distplot_AUC_summary_all-null.csv"
        outpath = os.path.join(outdir, outname)
        auc_df.to_csv(outpath)
        print(f"saved to {outpath}")

    return auc_df


## set 'flip=True' if pos_dist < dist_null is expected.
def get_roc(pos_dist, dist_null, n=None, flip=False):
    if n is None:
        n = len(pos_dist) + len(dist_null)
    elif n < 2:
        n = len(pos_dist) + len(dist_null)

    left_thresh = min(min(pos_dist.flatten()), min(dist_null.flatten()))
    right_thresh = max(max(pos_dist.flatten()), max(dist_null.flatten()))

    eps = (right_thresh - left_thresh)/n
    thresh = np.linspace( left_thresh-eps, right_thresh+eps, n+3)

    tpr = [None]*len(thresh)
    fpr = [None]*len(thresh)

    for i,t in enumerate(thresh):
        if flip:
            tpr[i] = np.mean( pos_dist < t )
            fpr[i] = np.mean( dist_null < t )
        else:
            tpr[i] = np.mean( pos_dist >= t )
            fpr[i] = np.mean( dist_null >= t )

    roc_curve = (fpr, tpr)
    auc = integrate(roc_curve)                          # AUC is senstive (complementary to) designation of positive vs. null distribution
    # auc = 1/2 + np.abs(1/2 - integrate(roc_curve))      # modifies AUC to be invariant w.r.t. designation of positive vs. null distribution

    return roc_curve, auc

def integrate(curve):
    dx = np.diff(curve[0])                      # width of x-step
    dy = curve[1][:-1] + np.diff(curve[1])/2    # average y-value within dx_i
    auc = (dx * dy).sum()                       # (Riemann) sum of trapezoidal areas dx_i*dy_i
    return np.abs(auc)


########################################################################################################################


# compute secondary statistics
########################################################################################################################
# add an empirical p-val to a dataframe containing only a single data-derived distance and its null counterparts
def _add_emp_pval(df, check_match=True, permtype=None, tail_type="all", corr_type="fdr", null_hi=None, null_lo=None, debug=False):
    df_data = df[df["datatype"] == "Data"]

    Wp_XY = df_data["Wp_XY"].to_numpy()
    if corr_type == "fdr":
        null_mask = df["datatype"].str.contains("Null")
        if permtype is not None:
            null_mask = null_mask & (df["permtype"] == permtype)
        df_null = df[ null_mask ]
        null_lo = df_null["Wp_XY"].to_numpy()
        null_hi = df_null["Wp_XY"].to_numpy()
        check_cols = [col for col in df.columns if col.startswith("X") or col.startswith("Y")]

        try:
            err_str =  f"data row and null rows are not of matching type: \n{[[col, df_data[col].unique(), df_null[col].unique()] for col in check_cols]}"
            assert all( [ df_data[col].unique() == df_null[col].unique() for col in check_cols ] ), err_str
        except AssertionError as err:
            print(f"failed with err: \n{err}")
            df_data.to_csv('debug/df_data_err.csv')
            df_null.to_csv('debug/df_null_err.csv')
            df.to_csv('debug/df_err.csv')
            print(f"offending dataframes saved to \'df_data_err.csv, df_null_err.csv, df_err.csv\' in \'{os.getcwd()}/debug\'")
            exit()

    err_msg = "extremal family-wise distributions must be provided for (\'fwe\') p-value correction: \nnull_hi={null_hi}\nnull_lo={null_lo}"
    assert (null_hi is not None) and (null_lo is not None), err_msg
    assert (len(null_hi) > 0) and (len(null_lo) > 0), err_msg

    if debug:
        print(f"Adding p-value of type {tail_type} with multiple comparison correction method {corr_type}")

    if tail_type == "all":
        tails = ["left", "right", "two-tailed"]
    else:
        tails = [tail_type]

    try:
        epsilon_lo = 1/len(null_lo)
        epsilon_hi = 1/len(null_hi)
        p_lo = 1 - np.mean(Wp_XY <= null_lo)
        p_hi = 1 - np.mean(Wp_XY >= null_hi)

        for tail in tails:
            if tail == "two-tailed":
                # compute p-value from two-tailed test against empirical CDF (enforcing inf(p)=1/N)
                empirical_pval = 2*min( max(epsilon_lo, p_lo), max(epsilon_hi, p_hi) )
            elif tail == "right":
                # compute p-value from 1-sided (right) test against empirical CDF (enforcing inf(p)=1/N)
                empirical_pval = min( max(epsilon_hi, p_hi), 1 - epsilon_hi )
            elif tail == "left":
                # compute p-value from 1-sided (left) test against empirical CDF (enforcing inf(p)=1/N)
                empirical_pval = min( max(epsilon_lo, p_lo), 1 - epsilon_lo )
            else:
                raise ValueError(f"Unrecognized p-value type {tail}")

            if permtype is None:
                df[f"{corr_type}-{tail}-pval"] = empirical_pval
            else:
                df[f"{corr_type}-{tail}-pval_{permtype}-null"] = empirical_pval

    except Exception as err:
        for tail in tails:
            print(f"p-value computation failed for correction type {corr_type} with tail type {tail}: \n{err}")
            df[f"{corr_type}-{tail}-pval"] = np.nan

    return df

def correct_pvals(pval_vec, verbose=True, low_thresh=0.01, high_thresh=0.05, corr_type="fdr"):
    low_count = np.count_nonzero(pval_vec < low_thresh)
    high_count = np.count_nonzero(pval_vec < high_thresh)
    if corr_type == "fdr":
        if verbose:
            print(f"correcting family of {len(pval_vec)} pvals")
            print(f"found {low_count} signficant (<{low_thresh}) pvals before correction")
            print(f"found {high_count} signficant (<{high_thresh}) pvals before correction")

        rmv_idx = np.isnan(pval_vec) + (pval_vec < 0)
        _, corr_pvals = fdrcorrection(pval_vec[ rmv_idx == False ])
        pval_vec[ rmv_idx == False ] = corr_pvals

        if verbose:
            low_count = np.count_nonzero(pval_vec < low_thresh)
            high_count = np.count_nonzero(pval_vec < high_thresh)
            print(f"found {low_count} signficant (<{low_thresh}) pvals after \'fdr\' correction")
            print(f"found {high_count} signficant (<{high_thresh}) pvals after \'fdr\' correction")
    elif corr_type == "fwe":
        if verbose:
            print(f"found {low_count} signficant (<{low_thresh}) pvals after \'fwe\' correction")
            print(f"found {high_count} signficant (<{high_thresh}) pvals after \'fwe\' correction")


    return pval_vec
########################################################################################################################


########################################################################################################################
default_pdiv_kwargs = {"axis":0, "lambda_":1, "sum_check":False, "ddof":-1}     # default initialization of _power_divergence kwargs

def make_chisq_summaries( value_set, gen_f_exp="by_rowsum", conflate_netmats=True, args=None, debug=False ):
    if args is not None:
        if args.verbose:
            print("Running alldata_df chi-squared.")
        alpha = args.alpha
    else:
        alpha = None

    # alldata_df = pd.DataFrame(data={k: v.flatten() for k,v in value_set.items()})
    alldata_df = pd.DataFrame(data={k: futils.triu_vals(v) for k,v in value_set.items()})
    alldata_df.dropna(axis='index', inplace=True)

    corr_type, perm_type = _parse_statvars(alldata_df.columns.values)

    # if True, do not distinigush between different types (partial, spatial, full) of network matrices when aggregating over feature type
    if conflate_netmats:
        alldata_df[["X_feature", "Y_feature"]]  = alldata_df[["X_feature", "Y_feature"]].map( lambda x: "NetMat" if "NMs" in x else x)
    
    alldata_df["sym_XY_ranks"] = [tuple(sorted(i)) for i in list(zip(alldata_df.X_rank, alldata_df.Y_rank))]
    alldata_df["sym_XY_metrics"] = [tuple(sorted(i)) for i in list(zip(alldata_df.X_metric, alldata_df.Y_metric))]
    alldata_df["sym_XY_features"] = [tuple(sorted(i)) for i in list(zip(alldata_df.X_feature, alldata_df.Y_feature))]
    alldata_df["sym_XY_featnums"] = [tuple(sorted(i)) for i in list(zip(alldata_df.X_feat_num, alldata_df.Y_feat_num))]
    alldata_df["sym_XY_parcellations"] = [tuple(sorted(i)) for i in list(zip(alldata_df.X_modality, alldata_df.Y_modality))]
    alldata_df["sym_XY_parcel_ranks"] = [tuple(sorted(i)) for i in list(zip(alldata_df.X_modality + alldata_df.X_rank.map(str), alldata_df.Y_modality + alldata_df.Y_rank.map(str)))]
    alldata_df["sym_XY_metric_ranks"] = [tuple(sorted(i)) for i in list(zip(alldata_df.X_metric + alldata_df.X_rank.map(str), alldata_df.Y_metric + alldata_df.Y_rank.map(str)))]

    mask_vars = [var for var in value_set.keys() if ("mask" in var) and ("two-tailed" not in var)]
    for var in mask_vars:
        alldata_df[var] = ~ alldata_df[var]
    newmask_names = {var: 
                     var.replace('left','Convergent').replace('right','Divergent').replace('-pval','').replace('_mask','') 
                     for var in mask_vars}
    alldata_df.rename( columns=newmask_names, inplace=True )
    mask_vars = list(newmask_names.values())
    alldata_df["Incomparable"] = alldata_df[mask_vars].apply(lambda x: not any(x), axis=1) 
    mask_vars.append("Incomparable")

    agg_vars = [var for var in alldata_df.columns.values if "sym" in var]

    if debug:
        print(f"Aggregation variables: \n{agg_vars}")
        print(f"Significance count variables: \n{mask_vars}")
        print(f"input dataframe: \n{alldata_df.filter(agg_vars + mask_vars, axis=1)}")

    chisq_results = general_chisquare_df( 
                                         alldata_df.filter(agg_vars + mask_vars, axis=1), 
                                         agg_vars, 
                                         mask_vars,
                                         gen_f_exp=gen_f_exp
                                         )
    if args is None:
        outdir = '.'
    else:
        outdir = args.output_dir
    outpath = os.path.join( outdir, f'chisq_results_{corr_type}-{perm_type}-alpha{alpha}.npy'.replace('0.','') )
    np.save( outpath, chisq_results, allow_pickle=True )

    return chisq_results



def stackplot_chisq( stackplot_df, gen_f_exp="by_rowsum", args=None ):
    if args is not None:
        if args.verbose:
            print("Running stackplot_df chi-squared.")
        alpha = args.alpha
    else:
        alpha = None

    agg_vars = ["Brain_Representation", "Parcellation", "Feature"]
    count_vars = ["Convergent", "Divergent", "Incomparable"]

    chisq_results = general_chisquare_df( 
                                         stackplot_df, 
                                         agg_vars=agg_vars, 
                                         count_vars=count_vars,
                                         gen_f_exp=gen_f_exp
                                         )
    if args is None:
        outdir = '.'
    else:
        outdir = args.output_dir
    outpath = os.path.join( outdir, f'stackplot_chisq_results_alpha{alpha}.npy'.replace('0.','') )
    np.save( outpath, chisq_results, allow_pickle=True )

    return None


def general_chisquare_df( df, agg_vars, count_vars, gen_f_exp="by_rowsum", debug=False, **pdiv_kwargs ):

    pdiv_kwargs = default_pdiv_kwargs | pdiv_kwargs

    chisq_results = []

    if debug:
        print("\ndebugging generalized chi-squared implmentation (for dataframes).")

    for i,var in enumerate(agg_vars):
        dropvars = copy.copy(agg_vars)
        dropvars.remove(var)

        if gen_f_exp is None:
            null_type= None
            null_df = None
        elif gen_f_exp == "by_rowsum":
            null_type="homogeneous Poisson"
            null_df = _generate_null_counts( df.filter(count_vars, axis=1) )
            null_df[agg_vars] = df[agg_vars]
            # enforce Poisson <-> chi-square approximation
            pdiv_kwargs["ddof"]=-1
            pdiv_kwargs["sum_check"]=False
        
        dropvars = copy.copy(agg_vars)
        dropvars.remove(var)
        agg_df = _contract_df(df.drop(columns=dropvars), agg_col=var)

        if any( np.mean( agg_df[count_vars], axis=0) < 5 ):
            pdiv_kwargs["lambda_"] = 2/3
        else:
            pdiv_kwargs["lambda_"] = 1

        if gen_f_exp == "by_col":
            sub_reslist = []
            
            countvar_pairs = list(itertools.permutations(count_vars, 2))
            for obs_col, null_col in countvar_pairs:
                f_obs = agg_df[obs_col].to_numpy(float)
                f_exp = agg_df[null_col].to_numpy(float)
                sub_reslist.append(
                        stats._stats_py._power_divergence(
                            f_obs=f_obs,
                            f_exp=f_exp,
                            **pdiv_kwargs
                            ))
            if debug:
                print(sub_reslist)
            chisq_stats = stats._stats_py.Power_divergenceResult(*list(zip(*sub_reslist)))
            obs_cols, null_cols = list(zip(*countvar_pairs))
            null_agg_df = (var, np.stack( (obs_cols, null_cols) ))
            obs_type = var
            null_type = "null hypothesis given by other distributions (see \'null_agg_df\' for order of observed-null column pairs)"
        else:
            obs_type = var
            f_obs=agg_df[count_vars].to_numpy(float)
            if null_df is None:
                f_exp=None
            elif gen_f_exp == "by_rowsum":
                null_agg_df = _contract_df(null_df.drop(columns=dropvars), agg_col=var)
                f_exp=null_agg_df[count_vars].to_numpy(float)

            ndims = len(np.squeeze(f_obs).shape)
            dimlist = [None] + list(range(ndims))

            sub_reslist = []
            for d in dimlist:
                pdiv_kwargs["axis"] = d
                sub_reslist.append(
                        stats._stats_py._power_divergence(
                        f_obs=f_obs,
                        f_exp=f_exp,
                        **pdiv_kwargs
                        ))
            chisq_stats = stats._stats_py.Power_divergenceResult(*list(zip(*sub_reslist)))
            pdiv_kwargs["axis"] = tuple(dimlist)


        chisq_results.append({
            "obs_type": obs_type,
            "null_type": null_type,
            "observed": agg_df,
            "expected_null": null_agg_df,
            "test_params": pdiv_kwargs,
            "statistics": chisq_stats
            })

    if debug:
        for res in chisq_results:
            for k,v in res.items():
                print(f"\tchi-squared results key \"{k}\" has value:\n{v}")

    return chisq_results


# computes null counts under the assumptions that the columns of df represent the counts of a column-wise homogeneous Poisson variable
def _generate_null_counts( df ):
    null_df = df.copy().astype(float)
    for i in range(len(df)):
        n = len(df.iloc[i])
        null_df.iloc[i] = [np.mean(df.iloc[i])]*n

    return null_df


# assumes that all non-sum columns are numeric/valid summation operands!
def _contract_df( df, agg_col="Feature", contraction=np.sum, debug=False):
    if debug:
        print(f"\ndebugging dataframe contraction (along repeated values).")
        print(f"input dataframe: \n{df}")
        print(f"summation column: {agg_col})")

    # variables forcetyped to 'str' to interact nicely with downstream text-search/plotting operators
    agg_vals = [str(i) for i in sorted(df[agg_col].unique().tolist())]

    if debug:
        print(f"unique values of the summation column: \n{agg_vals}")

    # initialize dictionary/labeled set of aggregated values
    agg_set = {}
    for val in agg_vals:
        agg_set[val] = np.sum( df[ 
                              df[agg_col].apply( lambda x: val==str(x) ) 
                              ].drop( columns=[agg_col] ), 
                          axis=0 )

    agg_df = pd.DataFrame(agg_set).T
    try:
        # numeric/quantitative/callable datatypes are restored if applicable
        agg_df[agg_col] = list(map(ast.literal_eval, agg_vals))
    except ValueError:
        agg_df[agg_col] = agg_vals
    
    if debug:
        # print(f"aggregated dictionary: \n{agg}")
        print(f"aggregated dataframe: \n{agg_df}")

    return agg_df

def _parse_statvars( varnames ):
    if any( [ 'fdr-left-pval' in var for var in varnames ] ):
        corr_type = 'fdr'
    elif any( [ 'fwe-left-pval' in var for var in varnames ] ):
        corr_type = 'fwe'
    else:
        corr_type = None

    if any( [ "subject" in var for var in varnames ] ):
        perm_type = "subject"
    elif any( [ "feature" in var for var in varnames ] ):
        perm_type = "feature"
    else:
        perm_type = None

    return corr_type, perm_type
