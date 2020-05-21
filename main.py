import cv2
import os, sys, time
import ffmpeg

import argparse
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)

#3rd party library
from utils import *


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="Output path to save video")
ap.add_argument("-f", "--frames", type=int, default=0, required=False,
	help="frames per second output video")

args = vars(ap.parse_args())

#print(args)

video_path = args['input']
saving_path = args['output']
frames_per_second = args['frames']

#################################################
##			       Loading Frames 		       ##
#################################################
if video_path[-3:] == "avi" or video_path[-3:] == "mp4":
    print("We are in: ", video_path)
    frames = Video_to_Frames(video_path)
    print("Done extracting frames from {}, number of frames is: {}".format(video_path, len(frames)) )
    print("-----------------------------------------------------------------------")


#Hint: TO-DO List:  - to plot the trajectory of the robot, we need at each frame to save it as png 
#					- then  read and add to a list, and then we use this list of frames to generate the video

#################################################
##			       Detecting Objects           ##
#################################################




#################################################
##			       Classification 		       ##
#################################################

centers, seen_frames = [], []
generating_frames = []
for frame in frames:
	#if frame.max() <= 1:					#Because resize() in plot trajectory, it rescales frames
	# frame = frame * 255
	center, _ = detect_arrow(frame)
	centers.append(center)
	seen_frames.append(frame)
	new_frame, _ = plot_trajectory2(frame, centers)
	generating_frames.append(new_frame)

#print("Centers: {}".format(centers))
print("Generated frames: ", len(generating_frames))




#################################################
##			       Saving Video 		       ##
#################################################
size = (generating_frames[0].shape[0],generating_frames[0].shape[1])
print("size: ", size)
vidwrite(saving_path, generating_frames, framerate=2, vcodec='libx264')