#!/bin/bash 
# Rotate video in 15-degree steps
for i in {0..345..15}
do
    echo "Rotation: "$i" degrees."
    ffmpeg -i ./robot_parcours_1.avi -c:v libx264 -crf 2 -vf "rotate="$i"*PI/180:ow='max(iw,ih)':oh='max(iw,ih)'" \
        ./robot_parcours_1_rotated_$i.mp4
    python3 ./main.py --input=./robot_parcours_1_rotated_$i.mp4 --output=./output_rotated_$i.mp4
done
