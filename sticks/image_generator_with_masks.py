#!/usr/bin/env

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import uuid
from scipy.ndimage.filters import gaussian_filter
#from blend_modes import blend_modes

#  from pascal_voc_writer import Writer

#%%

# GLOBAL PARAMETERS
L = 512  # image dimension
BACKGROUND = 80  # background lightness parameter
NOISE_LEVEL = 0.5  # background noise parameter
ROOT_PATH = '../datasets/sticks/'
print(ROOT_PATH)

#%%

def connect_points(img, p1, p2):
    """Connects two points by drawing a straight line between them.

    Parameters:
      img:  (N,N) numpy array       pixel values of the image
      p1:   tuple                   x and y coordinates
      p2:   tuple                   x and y coordinates

    Returns:
      img:  (N,N) numpy array       pixel values of the image
    """

    l = int(np.linalg.norm(p2 - p1))  # stick length

    # Determine the pixels between the two end points
    xs = np.linspace(p1[0], p2[0], l)
    ys = np.linspace(p1[1], p2[1], l)

    # Turn all stick pixels to lighter color
    for i in range(len(xs)):
        x, y = int(xs[i]), int(ys[i])
        # Prevent trying to color pixels outside of the image boundaries
        if x >= 0 and x < L and y >= 0 and y < L:
            img[y, x] = 240

    return img


    """Generates a reactangle around a stick. The two ends of the stick make up the two diagonal corners of the recatangle
        
    Parameters:
      img:  (N,N) numpy array       pixel values of the image
      p1:   (2,) numpy array       contains starting point of stick 
      p2:   (2,) numpy array       contains end point of stick
      file: file object            file that contains the annotations
    Returns:
      img:  (N,N) numpy array       pixel values of the image
    """
def generate_annot_rect(img,p1,p2,file):
    #Print out rectangles on img. Only needed for checking.
    #img=cv2.rectangle(img,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,0), 2)
    #Writing out starting and end points of the stick which make up the two diagaonal corners of the rectangle
    file.write(str(p1[0])+" "+str(p1[1])+" "+str(p2[0])+" "+str(p2[1])+"\n")
    
    return img


def generate_a_stick(img, MASKS_FOLDER_PATH):
    """Generates one stick at random.

    Parameters:
      img:  (N,N) numpy array       pixel values of the image
      file:  file object            file object that contains annotations
    Returns:
      img:  (N,N) numpy array       pixel values of the image
      bb:   (2, 2) list             bounding box coordinates for the added stick
    """

    imlen = len(img)  # image dimension

    # Stick max length between 1/50th and 1/10th of the image length
    lmax = np.random.randint(imlen / 50, imlen / 10)

    # Define the starting point for the stick
    # (upper left corner of a box containing the stick)
    p0 = np.random.randint((imlen - lmax, imlen - lmax))

    # End points of the stick
    p1 = p0 + np.random.randint((lmax, lmax))
    p2 = p0 + np.random.randint((lmax, lmax))
    
    #Call function generate_annot_rect() to create reactangle around stick as label.
    
    
    
    # Create some kinks on the stick by generating a random number
    # of additional points between the two end points
    n = np.random.randint(10)  # 10 kinks max
    p_list = [p1]  # list of all the points in the stick

    # Choose bend direction for the stick
    d_bend = np.random.randint(2)  # bend direction
    # Create a list of random additions to the chosen axis
    bends = np.random.randint(np.abs(p2[d_bend] - p1[d_bend]) + 1, size=(n + 2))
    bends[::-1].sort()  # sort the array in descending order

    for i in range(n):
        # Axis-wise distances between the current point and p2
        l = np.abs(p2 - p_list[i])
        # Axis-wise direction from the current point to p2
        # (multiplier -1 or 1 to tell if higher or lower)
        d = np.sign(p2 - p_list[i])

        # Create a new point and append to list.
        # Element-wise multiplication of d and l gives a new point on the line
        # between the current point and p2, and adding an element-wise
        # fluctuation shifts the point in random direction.
        # The shift is damped with a constant factor to reduce large kinks.
        p = p_list[i] + (d * l * np.random.rand(2) * 0.2).astype(int)

        #  # Add the bend
        #  if not (p[d_bend] + bends[i] < 0 or p[d_bend] + bends[i] > imlen):
        #      p[d_bend] += bends[i]
        p_list.append(p)

    # Finally append the last end poit
    p_list.append(p2)
    p_list = np.array(p_list)

    # Create the bounding box for the stick
    bb_x1 = np.min(p_list[:,0])
    bb_y1 = np.min(p_list[:,1])
    bb_x2 = np.max(p_list[:,0])
    bb_y2 = np.max(p_list[:,1])

    bb = [[bb_x1, bb_y1], [bb_x2, bb_y2]]

    # Draw the stick
    for i in range(len(p_list) - 1):
        img = connect_points(img, p_list[i], p_list[i+1])
        
    """
    Generate stick masks

    """
    
    img_gray = 80 * np.ones((L, L))
    
    # Draw the stick
    for i in range(len(p_list) - 1):
        img_mask = connect_points(img_gray, p_list[i], p_list[i+1])
        
    img_mask = img_mask.astype(np.uint8)
    img_mask = gaussian_filter(img_mask, sigma=1.2)
    img_mask = cv.cvtColor(img_mask, cv.COLOR_GRAY2RGB)
    
    ret, img_mask = cv.threshold(img_mask,100,255,cv.THRESH_BINARY)
    
    unique_filename = str(uuid.uuid4().hex)
    backtogray = cv.cvtColor(img_mask,cv.COLOR_RGB2GRAY)
    cv.imwrite(os.path.join(MASKS_FOLDER_PATH, '{}_'.format(i + 1) + unique_filename +".png"),backtogray)
    
    return img, bb
    
    #  img=generate_annot_rect(img,p1,p2,file)
    #  return img
    

def box_blur(img, X, Y, size, delta):

    newimage = img  # Copy of image
    imlen = len(img)  # Image dimension
    for x in range(X-delta, X+delta):
        for y in range(Y-delta, Y+delta):
            if x < 2 or y < 2 or x+2 > imlen or y+2 > imlen:  # Checking boundarires of image
                continue
            else:
                # Calculating average for 3x3 matrix
                sum = img[x - 1, y + 1] + img[x + 0, y + 1] + img[x + 1, y + 1] + img[x - 1, y + 0] + img[x + 0, y + 0] + img[x + 1, y + 0] + img[x - 1, y - 1] + img[x + 0, y - 1] + img[x + 1, y - 1] 
                newimage[x, y] = sum / 9

    return newimage


def generate_artifacts(img):
    """Generates the white spots seen on experimetal data, and the stronger vertical
    lines next to some of them.

    Parameters:
      img:  (N,N) numpy array       pixel values of the image

    Returns:
      img:  (N,N) numpy array       pixel values of the image
    """

    direction = np.random.choice(["left", "right"])  # choose either left or right

    imlen = len(img)  # image dimension

    # Keep track of used coordinates
    x_list = []
    y_list = []

    # Generate 5-10 smaller artifacts
    N = np.random.randint(5, 11)

    for i in range(N):
        size = np.random.randint(10, 200)  # artifact size

        # Select random coordinates for the artifact center
        x0 = int(np.random.randint(imlen - int(size / 10)))
        y0 = int(np.random.randint(imlen - int(size / 10)))
        x_list.append(x0)
        y_list.append(y0)

        # Generate points randomly near the selected coordinates and cumulate
        # lightness on them (the lighter the point is alread, the lighter it
        # will get)
        for j in range(size):
            p = np.random.rand()  # "lightness" added to the point

            # Random coordinates for an individual pixel near the artifact center
            while True:
                x = int(np.random.normal(x0, size / 5))
                y = int(np.random.normal(y0, size / 5))
                if x < 0 or x >= imlen or y < 0 or y >= imlen:
                    continue
                else:
                    break

            # Cumulate lightness to the point, with a constant weighting parameter
            img[x, y] += img[x, y] * p * 50
        #Gaussian blur approximating box blur algorithm is called
        img=box_blur(img,x,y,size,20)
            
    # Generate 2-5 bigger artifacts (put new artifacts on top of the existing
    # smaller ones)
    N = np.random.randint(2, 6)

    for i in range(N):
        size = np.random.randint(10, 200)  # artifact size

        # Randomly select one of the previous artifacts
        j = np.random.randint(len(x_list))

        # Define the center coordinates for the new added artifact
        while True:
            try:  # ugly way to prevent indices going out of borders
                # Deviate slightly from the previous artifact center to create
                # more uneven shapes
                x0 = x_list[j] + np.random.randint(100) - 100
                y0 = y_list[j] + np.random.randint(100) - 100
                break  # accept coordinates
            except IndexError:
                pass  # generate new coordinates

        for k in range(size):
            p = np.random.rand()  # "lightness" added to the point

            while True:
                try:  # Ugly way to prevent indices going out of borders
                    # Random coordinates for an individual pixel near the artifact center
                    x = int(np.random.normal(x0, size / 20))
                    y = int(np.random.normal(y0, size / 20))
                    img[x, y] += img[x, y] * p * 50
                    break  # accept coordinates
                except IndexError:
                    pass  # generate new coordinates
        #Gaussian blur approximating box blur algorithm is called
        delta=10
        img=box_blur(img,x,y,size,delta)
        delta=60
        img=box_blur(img,x,y,size,delta)
        #Generate line in front of artifact
        
        prob=np.random.rand()  #Probability determines whether line should be generated
        if prob<0.9:
            img=generate_artifact_line(img,x0,y0,direction)

    return img


"""Generates the white vertical lines that are in front or after artifact.

    Parameters:
      img:  (N,N) numpy array       pixel values of the image

    Returns:
      img:  (N,N) numpy array       pixel values of the image
    """
def generate_artifact_line(img,x,y,direction):
    delta=int(np.random.choice([0, np.random.rand() * 4])) #Thckness of artifact
    if direction=="left":
        img[x-delta:x+delta, 0:y] = 256
    else:
        img[x-delta:x+delta, y:len(img)] = 256
    
    return img 


def generate_stripes(img):
    """Generates the background vertical lines.

    Parameters:
      img:  (N,N) numpy array       pixel values of the image

    Returns:
      img:  (N,N) numpy array       pixel values of the image
    """

    imlen = len(img)  # image dimension
    vertical_nr=np.random.randint(10,50)
    # Go through vertical_nr number of vertical lines and generate a random line
    
    for i in range(vertical_nr):
        # Each end point of a line is randomly chosen to be either on the image
        # edge, or in a random poisition
        start = np.random.choice([0, np.random.randint(imlen)])
        stop = np.random.choice([np.random.randint(start, imlen), imlen])
        x=np.random.randint(0,imlen)
        # Each pixel between the two endpoints is colored darker.

        img[x, start:stop] -= np.random.choice([0, np.random.rand() * 100])
    return img


    """Generates the background white dots/spots (noise).

    Parameters:
      img:  (N,N) numpy array       pixel values of the image

    Returns:
      img:  (N,N) numpy array       pixel values of the image
    """
def background_white_dots(img):
    dots_nr=np.random.randint(100,500)  #Number of white dots
    for i in range(dots_nr):
        x=np.random.randint(1,len(img)-1) #Place of white dot
        y=np.random.randint(1,len(img)-1) #Place of white dot
        deltax=np.random.randint(0,5)  #Thickness of dot in x direction
        deltay=np.random.randint(0,5)  #Thickness of dot in y direction
        img[x-deltax:x+deltax,y-deltay:y+deltay]=np.random.randint(240,256) #white dot is added. 
    return img

    
#  def main():
#      
#      #The name of the file containg the annotations
#      filename="annotation"
#      
#      #Creating the file
#      file=open(filename,"w")
#      # Generate the image background
#      img = BACKGROUND * np.ones((L, L)) + np.random.randn(L, L) * NOISE_LEVEL


def save_annotations(bb_list, number):
    """Convert the bounding box coordinates into a Pascal VOC format and
    save them as an .xml file corresponding to an image.

    Parameters:
      bb_list:  (n, 2, 2) numpy array       list of the bounding box coordinates
      number:   int                         current image number
    """

    #  writer = Writer(os.path.join(ROOT_PATH, 'image_{}.jpg'.format(number)), L, L, 1)
    #  for bb in bb_list:
    #      writer.addObject('stick', bb[0, 0], bb[0, 1], bb[1, 0], bb[1, 1])

    #  writer.save('/home/hlappal/projects/ML-CSC-project/ImageAI-tutorial/sticks/train/annotations/annot_{}.xml'.format(number))

    pass

    
def main():
    # Generate 100 images
    N_images = 100              ################## Change! #################
    # N_train = int(N_images * 0.8)
    # N_test = N_images - N_train
    for i in range(N_images):
    #     if i < N_train:
    #         train_test_set = 'train'
    #     else:
    #         train_test_set = 'test'
    #     if i % 100 == 0:
    #         print("Generating {} images {}-{}/{}".format(train_test_set, i + 1, i + 100, N_images))
       

        #Create folder for images and masks
        FOLDER_NAME = str(uuid.uuid4().hex)
        #FOLDER_NAME = '{}'.format(i + 1)
        FOLDER_PATH = os.path.join(ROOT_PATH, FOLDER_NAME) 
        FOLDER_IMAGES = './images'
        FOLDER_MASKS = './masks'
        IMAGES_FOLDER_PATH = os.path.join(FOLDER_PATH, FOLDER_IMAGES)
        MASKS_FOLDER_PATH = os.path.join(FOLDER_PATH, FOLDER_MASKS)

        try:
            os.makedirs(FOLDER_PATH)
        except FileExistsError:
            # directory already exists
            pass

        try:
            os.makedirs(IMAGES_FOLDER_PATH)
        except FileExistsError:
            # directory already exists
            pass      
        
        try:
            os.makedirs(MASKS_FOLDER_PATH)
        except FileExistsError:
            # directory already exists
            pass       
        
        #print("Created a directory: " + str(IMAGE_NAME + ", in path: " + str(IMAGE_FOLDER)))
    
        # Generate the image background
        img = BACKGROUND * np.ones((L, L)) + np.random.randn(L, L) * NOISE_LEVEL
        background_sticks = np.zeros((L, L))
        
        plt.imshow(background_sticks, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        
        # Generate 50-200 sticks at random
        # N = np.random.randint(50, 201)
  
        bb_list = []  # List of bounding box coordinates for the generated sticks
        
        N_sticks_per_image = np.random.randint(10, 30)
        for k in range(N_sticks_per_image):
           
            N = 1
            for j in range(N):
                    
                img_sticks, bb = generate_a_stick(img, MASKS_FOLDER_PATH)
                bb_list.append(bb)
            

        bb_list = np.array(bb_list)

        # Generate the white dots
        img = generate_artifacts(img)

        # Generate the backround vertical lines
        img = generate_stripes(img)
        
        #Generate background white dots
        img = background_white_dots(img)
        
        # # Generate 50-200 sticks at random
        # N = 50
        # bb_list = []  # List of bounding box coordinates for the generated sticks
        # for j in range(N):
        #     img, bb = generate_a_stick(img,MASKS_FOLDER_PATH)
        #     bb_list.append(bb)

        # bb_list = np.array(bb_list)
        
        a = img/255
        b = img_sticks/255 # make float on range 0-1
        mask = a >= 0.5 # generate boolean mask of everywhere a > 0.5 
        ab = np.zeros_like(a) # generate an output container for the blended image 
        # now do the blending 
        ab[~mask] = (2*a*b)[~mask] # 2ab everywhere a<0.5
        ab[mask] = (1-2*(1-a)*(1-b))[mask] # else this 
        # Scale to range 0..255 and save
        img=(ab*255).astype(np.uint8) 
        

        # Do gaussian blurring on the whole image
        img = gaussian_filter(img, sigma=2)

        # Save the image and annotations
        # num = str(i + 1).zfill(5)
        # filename = os.path.join(
        #         ROOT_PATH,
        #         "sticks_{}".format(train_test_set),
        #         "image_{}.jpg".format(num)
        #         )
        # cv.imwrite(filename, img)
        
         
        #IMAGE_PATH = IMAGE_FOLDER + '/images/'
        #print(IMAGE_PATH)
        
        backtorgb = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
        cv.imwrite(os.path.join(IMAGES_FOLDER_PATH, FOLDER_NAME + '.png'),backtorgb)
        
        # bb_str = ' '
        # for bb in range(len(bb_list)):
        #     for i in range(2):
        #         for j in range(2):
        #             bb_str += str(bb_list[bb][i, j]) + ','
        #     bb_str += str(0) + ' '
        # bb_str += '\n'
        # bb_str = filename + bb_str
        # with open("sticks/sticks_{}.txt".format(train_test_set), "a") as f:
        #     f.write(bb_str)
        #  save_annotations(bb_list, i + 1)
    
        print(str(i) + "/" + str(N_images) + "-->" + str(FOLDER_NAME) + ".png")
    
    print("Done!")


if __name__ == "__main__":
    main()
