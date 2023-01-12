function [] = ext_ptseries(fname_path, parcel_path, out_path, subjlist_path)

addpath(genpath('/scratch/tyoeasley/brain_representations/src_MATLAB/fieldtrip'))


fname_path = convertStringsToChars(fname_path);
parcel_path = convertStringsToChars(parcel_path);
out_path = convertStringsToChars(out_path);
subjlist_path = convertStringsToChars(subjlist_path);

subj_IDs = readmatrix(subjlist_path);
n_subj = length(subj_IDs);
scanpars = {'1','2';'LR','RL'};

disp(parcel_path)
Parcellation = ft_read_cifti(parcel_path);
P = Parcellation.parcels;
n_parcels = max(P);

% output
disp(['Number of parcels in decomp: ' num2str(n_parcels)])
disp(['Pulling subject list from: ' subjlist_path])
disp(['Writing data to general path: ' out_path])

for i=1:n_subj
	[~, fname_path] = system(['/scratch/tyoeasley/brain_representations/check_subj.sh ',num2str(i)]);
	for j=1:2
		for k=1:2
			ID = num2str(subj_IDs(i));
			fname_path_ijk = sprintf(fname_path, ID, scanpars{1,j}, scanpars{2,k}, scanpars{1,j}, scanpars{2,k});
			% debug code:
			% disp(['pulling subject from ' fname_path_ijk])
				
			out_path_ijk = sprintf(out_path, ID, scanpars{1,j}, scanpars{2,k});

			try
				Data = ft_read_cifti(fname_path_ijk);
			catch
				% warning(['Subject ' ID ' not found in HCP_1200 release. Searching HCP_900...'])
				try
					fname_path_ijk = strrep(fname_path_ijk,'/HCP_1200/','/HCP_900/');
					Data = ft_read_cifti(fname_path_ijk);
				catch
					% warning(['Subject ' ID ' not found in HCP_900 release. Searching HCP_500...'])
					try
						fname_path_ijk = strrep(fname_path_ijk,'/HCP_900/','/HCP_500/');
						Data = ft_read_cifti(fname_path_ijk);
					catch
						warning(['Subject ' ID ' not found in ceph/hcpdb/archive/'])
						Data.dtseries = nan(n_subj,2);
						Data.brainstructure = nan(n_subj,1);
					end
				end
			end

			D = Data.dtseries;
			D_bs = Data.brainstructure;
			D(D_bs > 2, :) = [];

			parcel_data = zeros(n_parcels,size(D,2));

			for n=1:n_parcels
				if max(P) > size(D,1)
					% debugging code: if above condition is satisfied, code will fail 
					disp(['maximum index: ', max(P)])
					disp(['dimension size: ', size(D,1)])
				end
				parcel_signals = D(P == n,:);
				parcel_data(n,:) = mean(parcel_signals,1);
			end

			writematrix(parcel_data, out_path_ijk);
		end
	end
end
end
