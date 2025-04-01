import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import modules.preproc_module as mod

parser = argparse.ArgumentParser()
parser.add_argument("images",nargs="*",help="Add visual inspection photos for preprocessing")
args = vars(parser.parse_args())

images = []
# Load the images
for arg in args['images']:
    img = cv2.imread(arg)
    images.append(img)

# Initialize SIFT detector (or ORB can be used if needed)
sift = cv2.SIFT_create()

# Initialize BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) #NORM_L2

# Parameters for KNN matching
knn_ratio = 0.55
count = 55
score = 0
npts = 10
ransac = 7.0

# Load the first image and extract keypoints/descriptors
prev_image = images[0]
if prev_image is None:
    raise ValueError(f"Error loading {images[0]}")

prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

keypoint_image1 = cv2.drawKeypoints(prev_image, prev_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("keypoint_image1.png",keypoint_image1)

# Loop through the rest of the images
for i in range(1, len(images)):
    while npts >= 4: #score <= 0.25: # Could loop score or npts
        # Load the next image
        curr_image = images[i]
        
        if curr_image is None:
            print(f"Error loading {images[i]}")
            continue

        #Get source points for uncropped images
        height, width = prev_image.shape[:2]
        src_npts, dst_npts = mod.get_src_pts(bf, sift, knn_ratio,curr_image,prev_des,prev_kp, npts)
        
        '''
        #Use source points to crop images
        crop1, crop2 = mod.crop_to_match(prev_image, curr_image, src_npts, dst_npts, height, width)
        img1_name = args['images'][0].split('/')[-1]
        img2_name = args['images'][i].split('/')[-1]
        crop_image1_path = f'cropped/unwarped/crop_{count}_{img1_name}'
        crop_image2_path = f'cropped/unwarped/crop_{count}_{img2_name}'
        cv2.imwrite(crop_image1_path, crop1)
        cv2.imwrite(crop_image2_path, crop2)

        # Once saved, reload cropped images to warp
        c_prev_image = f'cropped/unwarped/crop_{knn_ratio}_{img1_name}'
        c_curr_image = f'cropped/unwarped/crop_{knn_ratio}_{img2_name}'
        c_prev_image = cv2.imread(c_prev_image)
        c_curr_image = cv2.imread(c_curr_image)

        c_height, c_width = c_prev_image.shape[:2]

        c_prev_gray = cv2.cvtColor(c_prev_image, cv2.COLOR_BGR2GRAY)
        c_prev_kp, c_prev_des = sift.detectAndCompute(c_prev_gray, None)

        #Get source points on cropped images
        c_src_npts, c_dst_npts = mod.get_src_pts(bf, sift, knn_ratio, c_curr_image,c_prev_des,c_prev_kp, npts)
        '''
        
        # Compute the homography matrices using RANSAC
        homography_matrix, mask = cv2.findHomography(src_npts, dst_npts, cv2.RANSAC, ransac)
 #       c_homography_matrix, c_mask = cv2.findHomography(c_src_npts, c_dst_npts, cv2.RANSAC, 5.0)

        if homography_matrix is not None:
                # Apply the homography to the current image

                transformed_image = cv2.warpPerspective(curr_image, homography_matrix, (width, height))
                #c_transformed_image = cv2.warpPerspective(c_curr_image, c_homography_matrix, (c_width, c_height))
                
                #c_score, c_mse_score = mod.alignment_score(c_prev_image,c_transformed_image)
                #print("Checking Crop scores...")
                #print("Image Crop score:", c_score)
                #print("Image MSE Crop score:", c_mse_score)
                
                img1_name = args['images'][0].split('/')[-1]
                img2_name = args['images'][i].split('/')[-1]
                #cropped_image1_path = f'homography/cropped/cropped_{knn_ratio}_{img1_name}'
                #cropped_image2_path = f'homography/cropped/cropped_{knn_ratio}_{img2_name}'

                #cv2.imwrite(cropped_image1_path, c_prev_image)
                #cv2.imwrite(cropped_image2_path, c_transformed_image)

                # Optionally save the transformed image
                img_name = args['images'][i].split('/')[-1]
                transformed_image_path = f'images/rembg/transformed_R0_T-rex_purp-{count}_{img_name}' #count or npts or ransac

                cv2.imwrite(transformed_image_path, transformed_image)
                print(f"Transformed image saved as {transformed_image_path}")
                print("Checking Score...")

                # Do note this only scores against the first image atm
                score, mse_score = mod.alignment_score(args['images'][0],transformed_image_path)
                print("Image Score: ",score)
                print("MSE Score: ", mse_score) # 0 = identical, < 100 = very similar, 100 < x < 1000 similar with noticable differences
                
                #ransac = ransac - 1.0
                npts = npts - 1
                knn_ratio = knn_ratio - 0.01
                count = count -1
                print("RANSAC: ",ransac)
                print("New Number of Points: ",npts)
                print("New threshold: ", knn_ratio)
                print("---------------------------")
        else:
                print(f"Homography could not be computed for {images[i-1]} and {images[i]}")
    #else:
    #        print(f"Not enough matches to compute homography for {images[i-1]} and {images[i]}")
    print("Found good alignment!")
        # Move to the next image. This would match each image to the one previously. ATM we are matching to the first image only
        #prev_image = curr_image
        #prev_kp = curr_kp
        #prev_des = curr_des 
