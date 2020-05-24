import numpy as np
import skimage
import matplotlib.pyplot as plt
import data_augmentation 
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
import cv2 as cv
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
import math

   
#----------------------------------------------------------------------
#finds the bounding boxes of significant symbols and extracts the symbols
#boxes : centroids of bounsing boxes
#images: extracted patches of the image that contain each symbol

def label_image(image):
    
    output = enhance_contrast(image)
    gray_im = skimage.color.rgb2gray(output)
    thresholded  = gray_im  < skimage.filters.threshold_otsu(gray_im )
    
    mask = closing(thresholded , square(10) )
         
    cleared = clear_border(mask)
    plt.imshow(cleared)
    
    labels, count = skimage.measure.label(cleared, connectivity=2, return_num=True)
    
   
    boxes = []
    symbols = []
    for region in regionprops(labels):
        if 1600 >= region.bbox_area >= 100:
            cx,cy = region.centroid
            cx, cy = int(cx),int(cy)
            #minx, miny, maxx, maxy
            boxes.append([cx,cy])
            new_symbol = gray_im[max(cx-20,0):min(cx+20,gray_im.shape[0]), max(cy-20,0):min(cy+20,gray_im.shape[1])]
            
            symbols.append( data_augmentation.rescale_down_sample(new_symbol,28,28) )
            
    
    print("Shapes properties: [minx, miny, maxx, maxy] \n", boxes)
    print("Count Objects: ", len(boxes))
    return boxes, symbols


#-----------------------------------------------------
#displays image with equally sized bounding boxes given box centroids and the image
def create_labeled_image(image,boxes,index):
    
#    
    for box in boxes:
        cx,cy = box[0],box[1]
        image = cv.rectangle(image, (max(cy-20,0),max(cx-20,0)),
                                    (min(cy+20,image.shape[1]), min(cx+20,image.shape[0])),           
                                    (200,50,0), 3)
        
    path = "labeled_frames/"+ "labeled" +str(index)+".png"
    
    cv.imwrite(path, cv.cvtColor(image, cv.COLOR_BGR2RGB) )  
    
#----------------------------------------------------------------------   
""" ADAPTED FROM LAB1 PART2: """
# Brightness and contrast adjustments: new_img = src * alpha + beta
# The parameters α>0 and β are often called the gain and bias parameters; 
# sometimes these parameters are said to control contrast and brightness respectively.

def enhance_contrast(src, alpha = 1.0, beta = 0):
    """
    src: input image
    Parameters α>0 and β are often called the gain and bias parameters; 
    output : img with same shape of the input output = alpha * src + beta
    """
    new_img = np.zeros(src.shape, src.dtype)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            for c in range(src.shape[2]):
                new_img[y, x, c] = np.clip(alpha * src[y, x, c] + beta, 0, 255)
    return new_img

#---------------------------------------------------------------------------
    #NOT COMPLETE
def update_trajectory(image, last_pos):
    arrow_mask =( image[:,:,0]> 100 )*(image[:,:,1]< 60)*(image[:,:,2]< 60)
    fig, ax = plt.subplots()
    ax.imshow(arrow_mask)
    regions =  regionprops(label(arrow_mask))
    if len(regions) > 1:     arrow = regions[np.argmax([reg.area for reg in regions])]
    elif len(regions) == 1:  arrow =regions[0]
    else: 
        print("arrow not found")
        return image, last_pos
#    cy,cx = arrow.centroid
#    
#    x1 = cx + math.cos(arrow.orientation) * 0.5 * arrow.minor_axis_length
#    y1 = cy - math.sin(arrow.orientation) * 0.5 * arrow.minor_axis_length
#    ax.plot((cx, x1), (cy, y1), '-g', linewidth=2.5)
#    ax.plot(cx, cy, '.g', markersize=15)
    
    
    
    y0, x0 = arrow.centroid
    orientation = arrow.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * arrow.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * arrow.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * arrow.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * arrow.major_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = arrow.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)
    
    
    
    
    
    
    return #[cy,cx], [y1,x1]




# TESTING

image = skimage.io.imread("robot_parcours_1_frames/frame0.jpg")


boxes, symbols = label_image(image)

create_labeled_image(np.array(image),boxes,0)

fig, ax = plt.subplots(1, len(boxes), figsize=(10, 6))
for i in range(len(boxes)):
   ax[i].imshow(symbols[i])








