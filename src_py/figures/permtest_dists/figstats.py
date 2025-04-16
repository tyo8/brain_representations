import numpy as np

# computes an empirical ROC curve (and AUC) given two data distributions.
########################################################################################################################
## set 'flip=True' if pos_dist < dist_null is expected.
def get_roc(pos_dist, dist_null, n=0, flip=False):
    if n < 2:
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
    auc = 1/2 + np.abs(1/2 - integrate(roc_curve))      # modifies AUC to be invariant w.r.t. designation of positive vs. null distribution

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
    assert null_hi is not None and null_lo is not None, err_msg

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

            df[f"{corr_type}-{tail}-pval"] = empirical_pval
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
