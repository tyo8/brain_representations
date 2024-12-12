#!/bin/sh

basedir="/scratch/tyoeasley/brain_representations/phom_analysis/full-scale-expmt"
subbasedir="${basedir}/resource_accounting"


all_cputimelines_path="${subbasedir}/used_cputime_out.txt"
all_cputime_path="${subbasedir}/used_cputime_vals.txt"
all_memlines_path="${subbasedir}/used_mem_out.txt"
all_mem_path="${subbasedir}/used_mem_vals.txt"

total_time=0
total_mem=0

if compgen -G $all_cputime_path >> /dev/null
then
	mv ${all_walltmime_path} ${all_cputime_path/vals/OLD_vals}
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

	method_cputime_path="${methods_dir}/${method_name}_used_cputime_vals.txt"
	method_mem_path="${methods_dir}/${method_name}_used_mem_vals.txt"

	echo "collecting resource usage information for ${method_name}..."
	if compgen -G $method_cputime_path >> /dev/null
	then
		mv ${method_walltmime_path} ${method_cputime_path/vals/OLD_vals}
	fi
	if compgen -G $method_mem_path >> /dev/null
	then
		mv ${method_mem_path} ${method_mem_path/vals/OLD_vals}
	fi

	for fpath in $fpath_list
	do
		# separately account resource usage data from failed and successful runs
		if $( grep -Fw "FAILED" $fpath )
		then
			all_cputimelines_path=${all_cputimelines_path/.txt/_FAILED.txt}
			all_cputime_path=${all_cputime_path/.txt/_FAILED.txt}
			all_memlines_path=${all_memlines_path/.txt/_FAILED.txt}
			all_mem_path=${all_mem_path/.txt/_FAILED.txt}
		fi
		cputime_line=$(grep -Fw "Used CPU Time" $fpath)
		hms=$( echo "$cputime_line" | cut -d: -f2-5 | cut -d' ' -f2 )
		if ! [ -z "$hms" ]:
		then
			IFS=: read h m s <<<"${hms%.*}"
			seconds=$(( 10#$s+10#$m*60+10#$h*3600 ))
		else
			seconds=0
		fi
		echo ${cputime_line} >> ${all_cputimelines_path}
		echo ${seconds} >> ${all_cputime_path}
		echo ${seconds} >> ${method_cputime_path}

		mem_line=$(grep -Fw "Max Mem Used" $fpath)
		memval=$( echo "$mem_line" | cut -d: -f3 | cut -d'(' -f2 | cut -d')' -f1 | cut -d. -f 1) 
		echo $mem_line >> ${all_memlines_path}
		echo $memval >> ${all_mem_path}
		echo $memval >> ${method_mem_path}

		let total_time="$total_time + $seconds"
		let total_mem="$total_mem + $memval"
	done
done

let total_GB="$total_mem/(1024*1024*1024)"
let total_hours="$seconds/3600"

echo "\n\n\n"
echo "Cumulative maximum RAM used: ${total_GB} GB"
echo "Cumulative CPU time used: ${total_hours} hours"
