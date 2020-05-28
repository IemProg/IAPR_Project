#!/bin/bash
echo "Bash version ${BASH_VERSION}..." 
# Rotate video in 15-degree steps
#for i in {0..345..15}
#	do
	    echo "--- Rotation: 45 degrees. ---- "
	    ffmpeg -i ./robot_parcours_1.avi -c:v libx264 -crf 2 -vf "rotate= 45 *PI/180:ow='max(iw,ih)':oh='max(iw,ih)'" \
	        ./robot_parcours_1_rotated_45.avi
	    python main.py --input=robot_parcours_1_rotated_45.avi --output=imad_rotated_45.avi
#	done
