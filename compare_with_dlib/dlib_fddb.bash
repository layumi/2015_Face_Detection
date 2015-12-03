#!/bin/bash
for((i=1;i<11;i++))do
	infilename=$(printf "/home/zzd/face-database/fddb/FDDB-fold-%02d.txt" $i)
	outfilename=$(printf "./fddb_result/fold-%02d-out.txt" $i)
	touch $outfilename
	k=1
	while read -r line;do
		./face_detection_fddb $line  >> $outfilename
		((k++))
	done < $infilename
done
