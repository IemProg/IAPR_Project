import os
import sys
import cv2

def Video_to_Frames(Video_file):
    head, tail = os.path.split(Video_file)
    output_location = head + "/" + tail[:-4]+ "_frames"
    os.mkdir(output_location)

    vidcap = cv2.VideoCapture(Video_file) # name of the video
    success, image = vidcap.read()
    count = 0
    while success: 
        cv2.imwrite(output_location+"/frame%d.jpg" % count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        #resized  = cv2.resize(image, (256, 256))
        #cv2.imwrite(output_location+"/frame%d.jpg" % count, resized)
        print('Read a new frame%d: '% count, success)
        count += 1


def resize(src, size, output_folder):
    head, tail = os.path.split(src)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    #print('Original Dimensions : ', img.shape)

    resized = cv2.resize(img, size)
    #print('Resized Dimensions ', tail, ": ", resized.shape)
    name = output_folder + "/" + tail
    cv2.imwrite(name, resized)

