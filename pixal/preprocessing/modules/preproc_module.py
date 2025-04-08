from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def get_equally_spaced_indices(n, start, end):
    # Generate n equally spaced indices from start to end
    indices = np.linspace(start, end, num=n, dtype=int)
    return indices

def get_score(pix1,pix2):
    # We are comparing against the first image, so we'll take pix1 as the true value
    pix1 = np.array(pix1)
    pix2 = np.array(pix2)
    t_values = []
    c_values = []
    
    t_values = pix1*2
    c_values = pix1+pix2

    for i in range(len(t_values)):
        for p in range(len(t_values[0])):
            if t_values[i][p] == 0:
                t_values[i][p] = 1
            if c_values[i][p] == 0:
                c_values[i][p] = 1
            
    f_values = np.array([a / b for a, b in zip(c_values, t_values)])

    # Calculate the average
    mean = np.mean(np.mean(f_values, axis=1),axis=0)
    
    return mean

def retrieve_grid_pixels(image, grid_size=(6, 6),offset=100):

    image = cv2.imread(image)
    
    if image is None:
        raise ValueError(f"Error loading image at {image}")

    # Get the height and width of the image
    height, width, _ = image.shape

        # Calculate valid height and width after the offset
    valid_height = height - offset
    valid_width = width - offset

    if valid_height < grid_size[0] or valid_width < grid_size[1]:
        raise ValueError("Image is too small for the specified grid size and offset.")

    # Calculate the spacing between pixels
    row_indices = np.linspace(offset, valid_height - 1, grid_size[0], dtype=int)
    col_indices = np.linspace(offset, valid_width - 1, grid_size[1], dtype=int)

    # Retrieve the pixels from the image
    pixels = []
    for r in row_indices:
        for c in col_indices:
            pixels.append(image[r, c])  # Append the pixel value

    return row_indices, col_indices

def mse(imageA, imageB):
    # Ensure images are of the same shape and type
    assert imageA.shape == imageB.shape, "Images must have the same dimensions."
    return np.mean((imageA - imageB) ** 2)

def crop_to_match(image1,image2, src_pts, dst_pts, height, width):

    margin = 200
    #Crop the overlapping region based on matched object locations
    # Define the cropping region using the common area
    x_min = max(0, np.min([src_pts[:,  0], dst_pts[:, 0]]).astype(int)-margin)
    y_min = max(0, np.min([src_pts[:,  1], dst_pts[:, 1]]).astype(int)-margin)
    x_max = min(width, np.max([src_pts[:,  0], dst_pts[:, 0]]).astype(int)+margin)
    y_max = min(height, np.max([src_pts[:,  1], dst_pts[:, 1]]).astype(int)+margin)
    '''
    # Ensure cropping coordinates are within the image bounds
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(width, int(x_max))
    y_max = min(height, int(y_max))
    '''
    # Crop the aligned image
    cropped_image1 = image2[y_min:y_max, x_min:x_max]
    cropped_image2 = image1[y_min:y_max, x_min:x_max]

    return cropped_image2, cropped_image1

def get_src_pts(bf, sift, knn_ratio,curr_image,prev_des,prev_kp, npts):
    # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors in the current image
        curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)
        
        # Apply KNN matching between the previous image and the current image
        matches = bf.knnMatch(prev_des, curr_des, k=2) #knnMatch

        good_matches = []
        for m, n in matches:
            if m.distance < knn_ratio * n.distance:
                good_matches.append(m)
        #print("Number of good matches found: ",len(good_matches))
        # Extract the matched keypoints
        if len(good_matches) > npts:  # At least 4 matches are required to compute the homography
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            src_len = len(src_pts)
            #print(src_len)
            #print(src_pts.shape)
            
            #ind = get_equally_spaced_indices(npts,0,src_len-1)

            #src_npts = []
            #dst_npts = []
            #for n in range(len(ind)):
            #    src_npts.append((src_pts[ind[n]][0][0],src_pts[ind[n]][0][1]))
            #    dst_npts.append((dst_pts[ind[n]][0][0],dst_pts[ind[n]][0][1]))
            
            src_npts = np.array(src_pts) #npts
            dst_npts = np.array(dst_pts) #npts

            return src_npts, dst_npts


def alignment_score(image1,image2):
    # This takes a set of pixels from both images and compares their values. 
    # The ideal scenario has all pixel values matching each other
    # This score tells us how close all chosen pixel values are to each other 
    # The idea is if all pixel values are almost exact, the image is aligned well

    # Let's get a grid of 9 pixels equally separated
    #row, col = retrieve_grid_pixels(img1)
    
    if isinstance(image1, str):
        image1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    if isinstance(image2, str):    
        image2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Compute SSIM and MSE between the two images
    mse_score = mse(image1,image2)
    score = ssim(image1, image2, win_size=3)
    '''
    #Load the images
    image1 = Image.open(img1)
    image2 = Image.open(img2)

    row, col = image1.size

    # Convert the images to HSV if it's not in that mode
    img1 = image1.convert('RGB')
    img2 = image2.convert('RGB')

    img1 = img1.load()
    img2 = img2.load()

    # Retrieve the pixels from the image
    pixels1 = []
    for r in range(row-1):
        for c in range(col-1):
            pixels1.append(img1[r,c])  # Append the pixel value for image1

    pixels2 = []
    for r in range(row-1):
        for c in range(col-1):
            pixels2.append(img2[r,c])  # Append the pixel value for image2

    score = get_score(pixels1,pixels2)
    '''
    return score, mse_score

