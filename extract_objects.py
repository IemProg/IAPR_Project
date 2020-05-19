import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

   
#----------------------------------------------------------------------
#finds the bounding boxes of significant symbols and extracts the symbols
#boxes : centroids of bounsing boxes
#images: extracted patches of the image that contain each symbol

def label_image(image):
    
    output = enhance_contrast(image)
    gray_im = skimage.color.rgb2gray(output)
    gray_im  = gray_im  < skimage.filters.threshold_otsu(gray_im )
    
    mask = closing(gray_im , square(10) )
         
    cleared = clear_border(mask)
    
    labels, count = skimage.measure.label(cleared, connectivity=2, return_num=True)
    
   
    boxes = []
    symbols = []
    for region in regionprops(labels):
        if 1600 >= region.bbox_area >= 100:
            cx,cy = region.centroid
            cx, cy = int(cx),int(cy)
            #minx, miny, maxx, maxy
            boxes.append([cx,cy])
            symbols.append( output[max(cx-20,0):min(cx+20,image.shape[0]), max(cy-20,0):min(cy+20,image.shape[1])])
    
    print("Shapes properties: [minx, miny, maxx, maxy] \n", boxes)
    print("Count Objects: ", len(boxes))
    return boxes, symbols


#-----------------------------------------------------
#displays image with equally sized bounding boxes given box centroids and the image
def display_labeled_image(image,boxes,index):
    
    
    plt.imshow(image)
    ax = plt.gca()
    
    for box in boxes:
        cx,cy = box[0],box[1]
        rect = mpatches.Rectangle((max(cy-20,0), max(cx-20,0)), min(cx+20,image.shape[0]) - max(cx-20,0), min(cy+20,image.shape[1]) - max(cy-20,0),
                                      fill=False, edgecolor='orange', linewidth=2)
        ax.add_patch(rect)
    path = "labeled_frames/"
    plt.savefig ( path + "labeled" +str(index)+".png")
    plt.show()
    
    
    
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
    #print(new_img.shape)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            for c in range(src.shape[2]):
                new_img[y, x, c] = np.clip(alpha * src[y, x, c] + beta, 0, 255)
    return new_img


# TESTING
image = skimage.io.imread("robot_parcours_1_frames/frame0.jpg")

boxes, symbols = label_image(image)

display_labeled_image(image,boxes,0)

fig, ax = plt.subplots(1, len(boxes), figsize=(10, 6))
for i in range(len(boxes)):
   ax[i].imshow(symbols[i])









