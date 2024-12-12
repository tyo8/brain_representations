
parent_pattern="within_"
detail_pattern="phom_data*/phom_out/phomXZ"

outfile="short_taglist.txt"
base_tagfile="subsample_tags.txt"
ntags=100

for i in $(ls ${parent_pattern}* -d) 
do 
	for tag in $(cat ${base_tagfile} | head -${ntags})
	do 
		num=$( ls ${i}/${detail_pattern}_${tag}.txt | wc -l ) 
		threshnum=$( ls ${i}/${detail_pattern/phom_out\/phom/dist_mtxs\/d}_${tag}.ldm | wc -l )
		if [[ "$num" -lt "$threshnum" ]]; 
		then 
			# echo "${i} (${num}): ${tag}"; 
			echo "${tag}" >> ${outfile}
		fi
	done
done

sort -u ${outfile} -o ${outfile}
