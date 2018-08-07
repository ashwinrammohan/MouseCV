#!/bin/bash

#ffmpeg -y -i 180807_01_under.mp4 -an -r 30 -pix_fmt rgb24 -vcodec tiff 180807_01_under/%06d.tif

for i in `seq 0 9`;
do
	tiffcp 180807_01_under/00$i*.tif 180807_01_under@000$i.tiff
	echo $i
done    

for i in `seq 10 30`;
do
	tiffcp 180807_01_under/0$i*.tif 180807_01_under@00$i.tiff
	echo $i
done 

echo All done