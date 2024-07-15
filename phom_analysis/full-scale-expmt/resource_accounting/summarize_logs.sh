#!/bin/sh

basedir="/scratch/tyoeasley/brain_representations/phom_analysis/full-scale-expmt"
subbasedir="${basedir}/resource_accounting"


all_walltime_path="${subbasedir}/used_walltime_vals.txt"
all_mem_path="${subbasedir}/mem_maxalloc_vals.txt"

if compgen -G $all_walltime_path >> /dev/null
then
	mv ${all_walltmime_path} ${all_walltime_path/vals/OLD_vals}
fi
if compgen -G $all_mem_path >> /dev/null
then
	mv ${all_mem_path} ${all_mem_path/vals/OLD_vals}
fi

methods_dir="${subbasedir}/per_method"
if ! test -d $methods_dir
then
	mkdir $methods_dir
fi


for logdir in $(ls "${basedir}/within_"*"/"*"/logs" -d )
do
	fpath_list=$( ls "${logdir}/shbsphom"*".out"* )

	method_name=$( basename "$( dirname $logdir )" )

	method_walltime_path="${methods_dir}/${method_name}_used_walltime_vals.txt"
	method_mem_path="${methods_dir}/${method_name}_mem_maxalloc_vals.txt"

	echo "collecting resource usage information for ${method_name}..."
	if compgen -G $method_walltime_path >> /dev/null
	then
		mv ${method_walltmime_path} ${method_walltime_path/vals/OLD_vals}
	fi
	if compgen -G $method_mem_path >> /dev/null
	then
		mv ${method_mem_path} ${method_mem_path/vals/OLD_vals}
	fi

	for fpath in $fpath_list
	do
		# only parse resource usage data from successful runs
		if ! $( grep -n $fpath -e "FAILED" )
		then
			hms=$( grep -n $fpath -e "Used Walltime" | cut -d: -f3-5 | cut -d' ' -f2 )
			if ! [ -z "$hms" ]:
			then
				IFS=: read h m s <<<"${hms%.*}"
				seconds=$(( 10#$s+10#$m*60+10#$h*3600 ))
			else
				seconds=0
			fi
			echo ${seconds} >> ${all_walltime_path}
			echo ${seconds} >> ${method_walltime_path}
			
			memval=$( grep -n $fpath -e "Max Mem Used" | cut -d: -f3 | cut -d'(' -f2 | cut -d')' -f1 ) 
			echo $memval >> ${all_mem_path}
			echo $memval >> ${method_mem_path}
		fi
	done
done
