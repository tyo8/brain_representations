import os
import sys
import dill
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 

src_dir = "/scratch/tyoeasley/brain_representations/src_py"
sys.path.append(src_dir)

def main(argvals):
    visualize_CCA_data(argvals[1],argvals[2])

def visualize_CCA_data(output_dir,fileloc):
    with open(fileloc,'rb') as fin:
        CCA_data = dill.load(fin)
    
    iter_viz(output_dir,CCA_data)


def iter_viz(output_dir,CCA_data):
    n_pairs = len(CCA_data)
    saveloc_regs = os.path.join(output_dir,'viz_regs.html')
    saveloc_corrs = os.path.join(output_dir,'viz_corrs.html')
    saveloc_comps = os.path.join(output_dir,'viz_comps.html')

    for i in range(n_pairs):
        ## debug code:
        # print(dir(CCA_data[i]))

        pairname = CCA_data[i].data_pair_name
        pairname = pairname.split('.')[0]
        pairname = pairname.replace('_and_',' and ')

        lambda_opts = CCA_data[i].lambda_opt
        comp_snorms = CCA_data[i].comp_norms_spectral
        comp_Fnorms = CCA_data[i].comp_norms_Frobenius
        corr_summs = CCA_data[i].corr_summaries
        visualize_regularization(saveloc_regs,pairname,lambda_opts,comp_snorms,comp_Fnorms,corr_summs)

        corr_dists = CCA_data[i].corr_dists
        visualize_corrs(saveloc_corrs,pairname,corr_summs,corr_dists)

        comp_sdists = CCA_data[i].comp_dists_spectral
        comp_Fdists = CCA_data[i].comp_dists_Frobenius
        if hasattr(CCA_data[i],'comp_corrs'):
            comp_corrs = CCA_data[i].comp_corrs
        elif hasattr(CCA_data[i],'comp_corr'):
            comp_corrs = CCA_data[i].comp_corr
        else:
            raise ValueError("CCA_Stability object has no attribute comp_corr/comp_corrs")

        ## debug code:
        # print(len(comp_corrs))
        # print(comp_corrs.shape)

        visualize_comps(saveloc_comps,pairname,comp_sdists,
                comp_snorms,comp_Fdists,comp_Fnorms,comp_corrs)


def visualize_regularization(saveloc,pairname,lambda_opts,comp_snorms,comp_Fnorms,corr_summs):
    n_iter = len(lambda_opts)
    lambda_opts = np.log10(lambda_opts)

    # NEXT: compute and title distribution histogram
    dist_counts,dist_bins = np.histogram(lambda_opts,density=False)
    dist_bins = 0.5*(dist_bins[:-1] + dist_bins[1:])
    dist_fig = px.bar(x=dist_bins, y=dist_counts, 
            title=r'$\text{'+ pairname +' regularizations over } n ='+str(n_iter)+r'\text{ iterations}$')

    dist_fig.update_layout(xaxis_title= r'$\text{Regularization values } (\log_{10}\lambda_0)$')
    with open(saveloc,'a') as fout:
        fout.write(dist_fig.to_html(full_html = False, include_mathjax = 'cdn', include_plotlyjs = 'cdn'))

    # NEXT: compute, title, and arrange the 2x2 grid of regularization vs. matrix norm plots (U,V)x(snorm,Fnorm)
    norm_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("U","V","U","V"))
    for i in range(len(comp_snorms)):
        norm_fig.add_trace(go.Scatter(x = lambda_opts, y = comp_snorms[i],mode='markers',showlegend=False), row=1, col=i+1)
        norm_fig.add_trace(go.Scatter(x = lambda_opts, y = comp_Fnorms[i],mode='markers',showlegend=False), row=2, col=i+1)
        norm_fig.update_xaxes(title_text=r'$\text{regularization value } (\log_{10}\lambda_0)$', 
                row=2, col=i+1)

    norm_fig.update_yaxes(title_text="Spectral Norm", row=1, col=1)
    norm_fig.update_yaxes(title_text="Frobenius Norm", row=2, col=1)
    norm_fig.update_layout(xaxis_title= r'$\text{Regularization values } (\log_{10}\lambda_0)$')

    norm_fig.update_layout(title_text = "Canonical Component Matrix norms vs. Regularization (" + pairname +")")
    with open(saveloc,'a') as fout:
        fout.write(norm_fig.to_html(full_html = False, include_mathjax = 'cdn', include_plotlyjs = 'cdn'))

    # NEXT: compute, title, and arrange the 1x3 grid of regularization vs. can. correlation plots (min, mean, max)
    corr_fig = make_subplots(
            rows=1, cols=len(corr_summs),
            subplot_titles=("min","geom. mean","max"))
    for i in range(len(corr_summs)):
        corr_fig.add_trace(go.Scatter(x = lambda_opts, y = corr_summs[i],mode='markers',showlegend=False), row=1, col=i+1)
        corr_fig.update_xaxes(title_text = r'$\text{regularization value } (\log_{10}\lambda_0)$', row=1, col=i+1)
    
    corr_fig.update_yaxes(title_text = "Pearson correlation", row=1, col=1)
    corr_fig.update_layout(title_text = "Canonical Correlation (min, gmean, and max)  vs. Regularization (" + pairname +")")
    with open(saveloc,'a') as fout:
        fout.write(corr_fig.to_html(full_html = False, include_mathjax = 'cdn', include_plotlyjs = 'cdn'))


def visualize_corrs(saveloc,pairname,corr_summs,corr_dist_mtx):
    n_iter = len(corr_summs[0])
    corr_dists = corr_dist_mtx[np.triu_indices(n_iter)]

    # NEXT: compute and title pairwise distance histogram
    dist_counts,dist_bins = np.histogram(corr_dists,density=False)
    dist_bins = 0.5*(dist_bins[:-1] + dist_bins[1:])
    dists_fig = px.bar(x=dist_bins, y=dist_counts, 
            title= pairname +" pairwise correlation L1 distances,  n=" + str(n_iter))
    dists_fig.update_layout(xaxis_title= r'$\lvert\lvert\rho_i\rvert - \lvert\rho_j\rvert\rvert_{i,j=1}^n$')
    with open(saveloc,'a') as fout:
        fout.write(dists_fig.to_html(full_html = False, include_mathjax = 'cdn', include_plotlyjs = 'cdn'))

    summs_fig = make_subplots(
            rows = 1, cols = len(corr_summs),
            subplot_titles=("min","geom. mean","max"))
    for i in range(len(corr_summs)):
        summ_counts,summ_bins = np.histogram(corr_summs[i],density=False)
        summ_bins = 0.5*(summ_bins[:-1] + summ_bins[1:])
        summs_fig.add_trace(go.Bar(x=summ_bins, y=summ_counts), row=1, col=i+1)
        summs_fig.update_xaxes(title_text = r'$\rho \text{ (Pearson)}$', row=1, col=i+1)

    summs_fig.update_layout(title_text = "Per-iteration summary statistics of canonical correlations (" + pairname + "), n="+str(n_iter))
    #NEXT: compute and title trio of correlation summary (min, mean, max) histograms
    #      -- maybe also look at *proportional* error in correlations later?
    with open(saveloc,'a') as fout:
        fout.write(summs_fig.to_html(full_html = False, include_mathjax = 'cdn', include_plotlyjs = 'cdn'))


def visualize_comps(saveloc,pairname,comp_sdists,comp_snorms,comp_Fdists,comp_Fnorms,comp_corrs):
    n_sets = len(comp_sdists)
    set_labels = ['U','V','W','S','T','Z']
    pairnames = [pairname]*n_sets
    for i in range(n_sets):
        pairnames[i] += " ("+set_labels[i]+")"

        ## debug code:
        # print(comp_corrs[i].shape)

        visualize_comp(saveloc,pairnames[i],comp_sdists[i],comp_snorms[i],comp_Fdists[i],comp_Fnorms[i],comp_corrs[i])
        

def visualize_comp(saveloc,pairname,comp_sdist_mtx,comp_snorms,comp_Fdist_mtx,comp_Fnorms,comp_corr_mtx):
    n_iter = len(comp_snorms)

    ## debug code:
    # print(comp_corr_mtx.shape)
    # print(comp_snorms.shape)
    # print(comp_sdist_mtx.shape)
    # print(comp_Fdist_mtx.shape)

    comp_corrs = comp_corr_mtx[np.triu_indices(n_iter)]
    comp_sdists = comp_sdist_mtx[np.triu_indices(n_iter)]
    comp_Fdists = comp_Fdist_mtx[np.triu_indices(n_iter)]


    corr_counts, corr_bins = np.histogram(comp_corrs, density=False)
    corr_bins = 0.5*(corr_bins[-1:] + corr_bins[1:])
    corrs_fig = px.bar(x=corr_bins, y=corr_counts, 
            title = "Correlations between canonical components of "+ pairname +", n=" + str(n_iter))
    corrs_fig.update_layout(xaxis_title=r'$\rho \text{ (Pearson)}$')
    # NEXT: compute and title histogram of component correlations
    with open(saveloc,'a') as fout:
        fout.write(corrs_fig.to_html(full_html = False, include_mathjax = 'cdn', include_plotlyjs = 'cdn'))

    # NEXT: compute and title histograms of component spectral norms and distances; 2x2 grid of (norm, dist)x(spec,Frob)
    normvals = [[comp_snorms,comp_Fnorms],[comp_sdists,comp_Fdists]]
    norms_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("CanComp Norms","CanComp Norms","Diff. Norms","Diff. Norms"))
    for i in range(2):
        for j in range(2):
            normcounts_ij,normbins_ij = np.histogram(normvals[i][j], density=False)
            normbins_ij = 0.5*(normbins_ij[:-1] + normbins_ij[1:])
            norms_fig.add_trace(go.Bar(x = normbins_ij, y = normcounts_ij), row=i+1, col=j+1)

        norms_fig.update_xaxes(title_text="Spectral Norm", row=2, col=1)
        norms_fig.update_xaxes(title_text="Frobenius Norm", row=2, col=2)
    norms_fig.update_layout(title_text= pairname +" Matrix and Difference Matrix Norms for Canonical Components, n="+str(n_iter))
    with open(saveloc,'a') as fout:
        fout.write(norms_fig.to_html(full_html = False, include_mathjax = 'cdn', include_plotlyjs = 'cdn'))

if __name__=="__main__":
    main(sys.argv)
