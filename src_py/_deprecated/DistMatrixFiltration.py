# receives name of csv file storing distance matrix
# computes and saves a persistence diagram

import gtda.homology as hml
import numpy as np

def comp_persdiag(fname_in,fname_out):

	distance_matrix = np.genfromtxt(fname_in,delimiter=",")

	# dimensions of homology groups
	hom_dims = [0, 1, 2]

	# give simplicial complex structure
	VR = hml.VietorisRipsPersistence(
		metric="precomputed", homology_dimensions=hom_dims
	)

	# compute persistence diagram (output as Nx[len(hom_dims)] array)
	persdiag = VR.fit_transform(distance_matrix[None,:,:])

	np.save(fname_out,persdiag)

	print("Persistence diagram computed from "+fname_in+" and saved to "+fname_out)
