import SubjSubjHomologies as SSH
import sys

SSH.comp_persdiag(sys.argv[1],sys.argv[2])
SSH.measure_phom_dists(sys.argv[2],fname_out=sys.argv[3])
SSH.comp_Betti_curves(sys.argv[2],fname_out=sys.argv[4])
SSH.comp_perst_imgs(sys.argv[2],fname_out=sys.argv[5])
