import os
import numpy as np
import pandas as pd
import seaborn as sns
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
                names = set(df["X_name"].to_numpy())
                if verbose:
                    print(f"In {names}, df_data has shape {df_data.shape}, df_null has shape {df_null.shape}")
                if debug:
                    ### debugging code ###
                    if df_null.empty:
                        print(f"empty df_null for nulltype=\'{null}\' pulled from: \n{df}\n")
                        print(f"df has datatypes={set(df.datatype)} and permtypes={set(df.permtype)}")
                        exit()
                    ### debugging code ###
                if df_data.empty:
                    # Will hold if H1 trivial for full data (nothing to bootstrap)
                    roc = (None, None)
                    auc = None
                    overlap = None
                elif df["PDX_diag"].unique() == 0:
                    roc = (None, None)
                    auc = 0
                    overlap = 1
                else:
                    try:
                        assert len(names) == 1, "Conflating distributions for more than one brain representation type."
                    except AssertionError as err:
                        print(f"More than one name found: {names}")
                        print(f"Offending dataframe written to: \n{os.getcwd()}/df_err.csv")
                        df.to_csv('df_err.csv')
                        exit()

                    if dist_type == "single":
                        nullnames = set(df_null["X_name"].to_numpy())
                        try:
                            assert names == nullnames, f"Comparing unmatched data and null distance distributions: \ndata={set(names)} \nnull={set(nullnames)}"
                        except ValueError as err:
                            print(f"failed for {set(names)} with error: \n{err}\n")
                            df_data.to_csv('df_data_err.csv')
                            df_null.to_csv('df_null_err.csv')
                            print(f"offending dataframes saved to \'df_data_err.csv, df_null_err.csv\' in \'{os.getcwd()}\'")
                            exit()


                    datavals = df_data[distvar].to_numpy()
                    nullvals = df_null[distvar].to_numpy()

                    # only counts as significant if subsamples are *closer* to original than null is
                    if distvar == "Wp_XY":
                        roc, auc = get_roc(datavals, nullvals, flip = flip)
                    elif distvar == "PDY_diag":
                        roc, auc_l = get_roc(datavals, nullvals, flip=True)
                        roc, auc_r = get_roc(datavals, nullvals)
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
    datarow = df[df["datatype"] == "Data"]

    Wp_XY = datarow["Wp_XY"].to_numpy()
    if corr_type == "fdr":
        nullrows = df[df["datatype"] == "Null"]
        if permtype is not None:
            nullrows = nullrows[ nullrows["permtype"] == permtype ]
        null_lo = nullrows["Wp_XY"].to_numpy()
        null_hi = nullrows["Wp_XY"].to_numpy()
        check_cols = [col for col in df.columns if col.startswith("X") or col.startswith("Y")]
        err_str =  f"data row and null rows are not of matching type: \n{[[col, set(datarow[col]), set(nullrows[col])] for col in check_cols]}"
        assert all( [ set(datarow[col]) == set(nullrows[col]) for col in check_cols ] ), err_str

    err_msg = "extremal family-wise distributions must be provided for family-wise error (\'fwe\') p-value correction"
    assert (null_hi is not None) and (null_lo is not None), err_msg

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
