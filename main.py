import cv2
import os, sys, time
import ffmpeg

import argparse
import matplotlib as mpl
import datetime
mpl.rc('figure', max_open_warning = 0)

#3rd party library
from utils import *

begin_time = datetime.datetime.now()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="Output path to save video")

ap.add_argument("-f", "--frames", type=int, default=2, required=False,
	help="frames per second output video")

args = vars(ap.parse_args())

#print(args)

video_path = args['input']
saving_path = args['output']

FPS = args['frames']

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
## Detecting Centers, Boxes of the Red arrow   ##
#################################################
arrow_centers = []
arrow_boxes = []
for frame in frames:
    arrow_center, center_box = detect_arrow(frame)
    arrow_centers.append(arrow_center)
    arrow_boxes.append(center_box)

#################################################
##			       Detecting Objects           ##
#################################################

# Note: Real_boxes and real_centers should have the same order
_, real_boxes, real_centers = AllInfoFromFrame0(frames[0], arrow_centers[0])
objects = extract_signs(real_boxes, frames[0])
print("Number of Detected Real_Boxes is: {}".format(len(real_boxes)))
print("Shape of Extacted signs is: {}".format(objects.shape))

# We initialise a list of False when the robot passed the box, we change the correspond boolean value to True
passed = []
for box in real_boxes:
	passed.append(False)

#################################################
##			       Classification 		       ##
#################################################

#mydictionary = classify(real_boxes, objects)
mypredictions = {}

#Just to test the functionning of the script, since no classifier is provided
#Note: predictio should has the same order as the boxes

for i, box in enumerate(real_boxes):
	mypredictions[i] = i	

print("mypredictions: ", mypredictions)
#################################################
##			       Editing Frames 		       ##
#################################################
centers, seen_frames = [], []
generating_frames = []

for k, frame in enumerate(frames):
	print("---- We are in frame: {}".format(k))
	#if frame.max() <= 1:					#Because resize() in plot trajectory, it rescales frames
	# frame = frame * 255
	#center, _ = detect_arrow(frame)
	#centers.append(center)
	centers.append(arrow_centers[k])
	seen_frames.append(frame)

	# We need to check here if the center added does belong to operator/digit box 
	# If it is True, we need to write the equation
	for index, box in enumerate(real_boxes):
		if intersect(arrow_centers[k], real_centers[index]):
			print("\tI'm in box: {}: ".format(box))
			label_detected = mypredictions[index]
			print("\tLabel: {}".format(label_detected))
			passed[index] = True
			#Plot the the detected digit/operator on the frame

	new_frame, _ = plot_trajectory2(frame, centers, real_boxes, mypredictions, passed)
	generating_frames.append(new_frame)

	#if the the label detected is equal means: STOP
	if label_detected = "=":
		break

#print("Centers: {}".format(centers))
print("Generated frames: ", len(generating_frames))



#################################################
##			       Saving Video 		       ##
#################################################
size = (generating_frames[0].shape[0],generating_frames[0].shape[1])
#print("size: ", size)
vidwrite(saving_path, generating_frames, framerate=FPS, vcodec='libx264')

print("Script executed in: ", datetime.datetime.now() - begin_time)