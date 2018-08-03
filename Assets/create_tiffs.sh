#!/bin/bash

#ffmpeg -y -i 180713_12_under.mp4 -an -r 30 -pix_fmt rgb24 -vcodec tiff 180713_12_under/%06d.tif

for i in `seq 0 9`;
do
	tiffcp 180713_12_under/00$i*.tif 180713_12_under@000$i.tiff
	echo $i
done    

for i in `seq 10 30`;
do
	tiffcp 180713_12_under/0$i*.tif 180713_12_under@00$i.tiff
	echo $i
done 

echo All done