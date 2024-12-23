import os
import csv
import datetime
import numpy as np
import gtda
from gtda import plotting
import matplotlib
from matplotlib import pyplot as plt

def_labelset = ["NM_PRO","Maps_PRO","Amps_PRO","NM_MMP","Amps_ICA"]

def export_PHdgms(subj_dists_name,phoms_array_name):
    phoms_array = np.load(phoms_array_name) # load array of persistence diagrams
    rep_dist_locs = get_lines(subj_dists_name) # load lines of .csv file containing names of subj-subj distance matrices
    dtypeset = [parse_datatype(i) for i in rep_dist_locs]
    
    n_dgms = phoms_array.shape[0] # number of persistence diagrams in phoms_array
    n_reps = len(rep_dist_locs) # number of representations for which subj-subj dist mtxs were stored

    date_label = datetime.date.today().strftime("%Y%b%d") # current date in YYYY-mmm-dd format
    fig_dir = os.path.dirname(phoms_array_name)+"/PDs"
    lims = [0,.9]

    dgm_fname_type = os.path.join(fig_dir, PHdgms, 'PH_%s_' + date_label + '.html')  # filepath to saved diagram (saved as HTMl)
    dgm_fnamelist = [dgm_fname_type % dtype for dtype in dtypeset]    # find representation and feature type of diagram data
    titleset = ["Subject-Subject PD for %s" % dtype for dtype in dtypeset]
    plot_PHdgms(phoms_array, dgm_fnamelist, titleset, lims=lims)


def plot_PHdgms(phom_list, fnamelist, titleset, lims=[0, 1.05]):
    n_dgms = len(phom_list)
    for i in range(n_dgms):
        pers_dgm = plot_PHdgm(phom_list[i], lims=lims, title=titleset[i])
        if not os.path.isdir(os.path.dirname(fnamelist[i])):
            os.makedirs(os.path.dirname(fnamelist[i]))
            print("Warning: created directory " + os.path.dirname(fnamelist[i]))
        pers_dgm.write_html(fnamelist[i])


def plot_PHdgm(phom_data, lims=[0, 1.05], title='Persistence Diagram'):
    pers_dgm = gtda.plotting.plot_diagram(phom_data)
    pers_dgm.layout.xaxis.range = lims
    pers_dgm.layout.yaxis.range = lims
    pers_dgm.update_layout(title = title)
    return pers_dgm


def parse_datatype(fname):
    fsplit = fname.split("subj_dists_")
    dtype_w_ext = fsplit[-1].split(".")
    dtype = dtype_w_ext[0]
    return dtype



def get_lines(csv_fname):
    with open(csv_fname) as fin:
        LL = list(csv.reader(fin))

    list_of_lines = list(map(''.join,LL))
    return list_of_lines


def export_dists(fig_dir,phom_dists=np.asarray([]), phom_dists_fname='',fname_prefix="W2_dists_H"):
    os.makedirs(fig_dir, exist_ok = True)

    if not phom_dists.any():
        assert phom_dists_fname, 'If no persistence barcode array is given, a filename to one must be given instead'
        if phom_dists_fname:
            phom_dists = np.load(phom_dists_fname)

    n_dims = phom_dists.shape[-1]
    for d in range(n_dims):
        D = phom_dists[:,:,d]
        img_name = os.path.join(fig_dir, fname_prefix+str(d))
        save_w_colorbar(D,img_name)


def save_w_colorbar(img_data,img_name):
    fig = plt.figure(figsize = (3,3),dpi=600) # Your image (W)idth and (H)eight in inches
    
    # Stretch image to full figure, removing "grey region"
    plt.subplots_adjust(left = 0, right = 0.8, top = 1, bottom = 0)

    # Show the image
    im = plt.imshow(img_data)
    # Set colorbar position in fig
    pos = fig.add_axes([0.81,0.15,0.02,0.7]) 
    # Create the colorbar
    fig.colorbar(im, cax=pos)
    plt.savefig(img_name)
    plt.close()


def export_bcurves(fig_dir, betti_curveset=np.asarray([]), betti_curveset_name='', 
        parent_figlist=[], parent_axlist=[], colors=np.asarray([]), linestyle=[],
        fname_prefix="Bcurve_H", title_suffix='', labelset=def_labelset):
    os.makedirs(fig_dir,exist_ok = True)

    ## debug code:
    # print("Betti curves saved to directory: " + fig_dir)

    if not betti_curveset.any():
        if betti_curveset_name:
            betti_curveset = np.load(betti_curveset_name)

    betti_curves = betti_curveset[:-1,:,:]  # unpack Betti curves from curveset
    filt_params  = betti_curveset[-1,:,:]   # unpack filtration parameter samplings from curveset

    n_reps = betti_curves.shape[0]          # number of brain representations
    n_dims = betti_curves.shape[1]          # number of homology dimensions

    x_label = "Filtration parameter, 0 < t < 1"
    y_label = "Betti number"
    if not title_suffix:
        title_suffix = str(n_reps)+" Brain Representations"

    if parent_figlist:
        figlist = parent_figlist
        axlist = parent_axlist
    else:
        figlist = [None for i in range(n_dims)]
        axlist = [None for i in range(n_dims)]

    ## debug code:
    # print("current list of figures: ", parent_figlist)
    for i in range(n_dims):
        if parent_figlist:
            fig = figlist[i]
            ax = axlist[i]
        else:
            fig,ax = plt.subplots()


        Dcurves = betti_curves[:,i,:]       # Betti curve in i-th homology dimension
        filt_vals = filt_params[i,:]        # sampled filtration paramater values in i-th hom. dimension

        plot_path = os.path.join(fig_dir,fname_prefix+str(i)+".png")
        title = "Betti curves in H" + str(i) + " for\n" + title_suffix

        if len(colors) == 0:
            colors = plt.cm.rainbow(np.linspace(0,1,n_reps))
        if not linestyle:
            linestyle = '-'

        for j in range(n_reps):
            ax.plot(filt_vals, Dcurves[j,:].transpose(), linestyle, c=colors[j], label=labelset[j])

        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        fig.savefig(plot_path, dpi=600)
        figlist[i] = fig
        axlist[i] = ax

    return figlist, axlist



def export_imgs(perst_imgs_name,fig_dir,fname_prefix="/PerstImg_BR"):
    os.makedirs(fig_dir,exist_ok=True)

    date_label = datetime.date.today().strftime("%Y%b%d") # current date in YYYY-mmm-dd format
    perst_imgs  = np.load(perst_imgs_name)

    n_reps      = perst_imgs.shape[0]
    n_dims      = perst_imgs.shape[1]

    for i in range(n_reps):
        for j in range(n_dims):
            img_path = fig_dir + fname_prefix + str(i) + "_H" + str(j) + "_" + date_label + ".png"
            perst_img = perst_imgs[i,j,:,:]
            save_w_colorbar(perst_img,img_path)
