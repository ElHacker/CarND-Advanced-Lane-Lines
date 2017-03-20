import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
import os.path


CONST_CAMERA_CALIB_PICKLE_FILE_PATH = 'output_images/dist_pickle.p'

def getObjectAndImagePoints():
    """
    Extract object points and image points for camera calibration.
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints, imgpoints


def prepareCamera(objpoints, imgpoints):
    """
    Calibrates camera, calculates distortion coefficients and test undistortion
    on an image
    :objpoints: the object points extracted from the calibration images.
    :imgpoints: the image points extracted from the calibration images.
    """

    # Test undistortion on an image
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/calibration_undist.jpg',dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(CONST_CAMERA_CALIB_PICKLE_FILE_PATH, "wb" ))
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # Visualize undistortion
    visualizeImages(img, dst)

def undistortImage(img):
    """
    Undistorts an image. Loads camera calibration params from pickle file
    :img: image to undistort
    returns undistorted image
    """
    dist_pickle = pickle.load(open(CONST_CAMERA_CALIB_PICKLE_FILE_PATH, "rb"))
    mtx =  dist_pickle['mtx']
    dist = dist_pickle['dist']
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img


def absSobelThresh(img, orient='x', thresh_min=0, thresh_max=255, thresh_limits=(0, 255)):
    """
    Define a function that applies Sobel x or y,
    then takes an absolute value and applies a threshold.
    :img: the image to apply sobel to. The image must be in the RGB color space.
    :orient: in which direction apply sobe.
    :thresh_limits: min and max thresholds to generate the binary image.
    """
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    orientX, orientY = (1, 0) if orient == 'x' else (0, 1)
    sobel = cv2.Sobel(gray, cv2.CV_64F, orientX, orientY)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresh_limits[0]) & (scaled_sobel < thresh_limits[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = sxbinary
    return binary_output

def magSobelThresh(img, sobel_kernel=3, thresh_limits=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    grad_magnitude = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*grad_magnitude/np.max(grad_magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_limits[0]) & (scaled_sobel <= thresh_limits[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def dirSobleThresh(img, sobel_kernel=3, thresh_limits=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresh_limitsolds are met
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= thresh_limits[0]) & (dir_sobel <= thresh_limits[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def combineSobel(grad_binary_x, grad_binary_y, mag_binary, dir_binary):
    combined = np.zeros_like(dir_binary)
    combined[((grad_binary_x == 1) & (grad_binary_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def hls_select(img, thresh_limits=(0, 255)):
    """
    Define a function that thresholds the S-channel of HLS
    Use exclusive lower bound (>) and inclusive upper (<=)
    """
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh_limits[0]) & (S <= thresh_limits[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def warpImage(img):
    """
    Compute and apply perpective transform.
    :img:  the image to warp
    """
    img_size = (img.shape[1], img.shape[0])
    # Points order:
    # left_upper
    # left_lower
    # right_upper
    # right_lower
    src = np.float32([
        [img_size[0] / 2 - 65, img_size[1] / 2 + 100],
        [img_size[0] / 6 - 10, img_size[1]],
        [img_size[0] * 5 / 6 + 60, img_size[1]],
        [img_size[0] / 2 + 65, img_size[1] / 2 + 100]])
    dst = np.float32([
        [img_size[0] / 4 - 30, 0],
        [img_size[0] / 4, img_size[1]],
        [img_size[0] * 3 / 4, img_size[1]],
        [img_size[0] * 3 / 4, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped_img, src, dst


def visualizeImages(original_img, modified_img, modified_title='Modified Image', is_modified_img_gray=False):
    """
    Show original image along with the modified version.
    :original_img: The original image with camera distortion.
    :modified_img: The undistorted version of the original image.
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=30)
    if (is_modified_img_gray):
        ax2.imshow(modified_img, cmap="gray")
    else:
        ax2.imshow(modified_img)
    ax2.set_title(modified_title, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def defineFlags():
    parser = argparse.ArgumentParser(description='Detects lane lines.')
    parser.add_argument('--force-calibration', action='store_true')
    return parser.parse_args()


def main():
    args = defineFlags()
    if (args.force_calibration or not os.path.isfile(CONST_CAMERA_CALIB_PICKLE_FILE_PATH)):
        objpoints, imgpoints = getObjectAndImagePoints()
        prepareCamera(objpoints, imgpoints)
    # image = undistortImage(mpimg.imread('test_images/straight_lines1.jpg'))
    image = mpimg.imread('test_images/straight_lines1.jpg')
    grad_binary_x = absSobelThresh(image, orient="x", thresh_limits=(20, 100))
    grad_binary_y = absSobelThresh(image, orient="y", thresh_limits=(20, 100))
    mag_binary = magSobelThresh(image, thresh_limits=(30, 100))
    dir_binary = dirSobleThresh(image, sobel_kernel=15, thresh_limits=(0.7, 1.3))
    combined = combineSobel(grad_binary_x, grad_binary_y, mag_binary, dir_binary)
    # visualizeImages(image, combined, 'Combined', True)
    hls_binary = hls_select(image, thresh_limits=(170, 255))
    # visualizeImages(image, hls_binary, 'S channel', True)

    # Combine the Sobel on X with the HLS thresholded in S channel.
    combined_grad_binary_x_with_hls_binary = np.zeros_like(grad_binary_x)
    combined_grad_binary_x_with_hls_binary[(grad_binary_x == 1) | (hls_binary == 1)] = 1
    visualizeImages(image, combined_grad_binary_x_with_hls_binary, 'Sobel X gradient with S channel', True)

    # Warp the image.
    # warped_image = warpImage(combined_grad_binary_x_with_hls_binary)
    warped_image, src_points, dst_points = warpImage(image)
    visualizeImages(image, warped_image, 'Warped image', True)

main()
