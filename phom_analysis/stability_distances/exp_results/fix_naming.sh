rm -rf All_vs_*/X_*/X_*_ztrans*
rm -rf within_*/permtesting/X_*/X_*_ztrans*

for i in All_vs_*/X_*/*_ztrans*.csv
do
	mv ${i} ${i//"_ztrans"/"-ztrans"}
done

for i in All_vs_*/X_*_ztrans*
do
	mv ${i} ${i//"_ztrans"/"-ztrans"}
done

for i in All_vs_*/X_*/*_ztrans*.csv
do
	mv ${i} ${i//"_ztrans"/"-ztrans"}
done

rmdir All_vs_*/X_*_ztrans*

for i in within*/permtesting/X_*/*_ztrans*.csv
do
	mv ${i} ${i//"_ztrans"/"-ztrans"}
done

for i in within*/permtesting/X_*_ztrans*
do
	mv ${i} ${i//"_ztrans"/"-ztrans"}
done

for i in within*/permtesting/X_*/*_ztrans*.csv
do
	mv ${i} ${i//"_ztrans"/"-ztrans"}
done

rmdir within*/permtesting/X_*_ztrans*

for i in within*/subsampling/*_ztrans*.csv
do
	mv ${i} ${i//"_ztrans"/"-ztrans"}
done

