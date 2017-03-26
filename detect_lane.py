import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
import os.path
from moviepy.editor import VideoFileClip
from line import Line


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
        [img_size[0] / 2 - 60, img_size[1] / 2 + 100],
        [img_size[0] / 6 - 10, img_size[1]],
        [img_size[0] * 5 / 6 + 50, img_size[1]],
        [img_size[0] / 2 + 60, img_size[1] / 2 + 100]])
    dst = np.float32([
        [img_size[0] / 4, 0],
        [img_size[0] / 4, img_size[1]],
        [img_size[0] * 3 / 4, img_size[1]],
        [img_size[0] * 3 / 4, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped_img, src, dst, M, Minv


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
    parser.add_argument('--video', action='store_true')
    return parser.parse_args()


def drawLines(image, points, color=[255, 0, 0], thickness=4):
    """
    Draw the left and right lane lines.
    :points: has the 4 points required to draw the two lane lines.
    """
    image = cv2.line(image, tuple(points[0]), tuple(points[1]), color, thickness)
    image = cv2.line(image, tuple(points[2]), tuple(points[3]), color, thickness)
    return image


def generateHistogram(binary_img, plot=False):
    """
    Take a histogram of the bottom half of a binary image
    """
    histogram = np.sum(binary_img[binary_img.shape[0]/2:,:], axis=0)
    if (plot):
        plt.plot(histogram)
        plt.show()
    return histogram

def slideWindowsFitPolynomial(binary_warped, plot=False):
    """
    Finds lane pixels by Histogram and sliding window.
    Video: https://www.youtube.com/watch?v=siAMDK8C_x8
    """
    histogram = generateHistogram(binary_warped)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    ### VISUALIZE
    # Generate x and y values for plotting
    if (plot):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_lane_inds, right_lane_inds, left_fit, right_fit

def fitPolynomialAroundLinePositions(binary_warped, left_line, right_line, plot=False):
    """
    Fits a polynomial function based on previously calculated line positions on the method
    slideWindowsFitPolynomial.
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
            & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
            & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    ### VISUALIZE
    if (plot):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_lane_inds, right_lane_inds, left_fit, right_fit

def calculateRadiusOfCurvatureAndCenterInWorldSpace(image, binary_warped, left_lane_inds, right_lane_inds):

    _, leftx, lefty, rightx, righty = getNonzeroPositions(binary_warped, left_lane_inds, right_lane_inds)

    # Define conversions in x and y from pixels space to meters
    # lane is about 30 meters long and 3.7 meters wide.
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty)*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(righty)*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Calculate distance from center
    center_of_lane = ((rightx[-1] - leftx[-1]) / 2 + leftx[-1]) * xm_per_pix
    center_of_image = (image.shape[1] / 2) * xm_per_pix
    dist_from_center = center_of_image - center_of_lane
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm', dist_from_center, 'm')

    return left_curverad, right_curverad, dist_from_center


def getNonzeroPositions(binary_warped, left_lane_inds, right_lane_inds):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return ploty, leftx, lefty, rightx, righty


def drawLane(image, binary_warped, left_fit, right_fit, Minv, left_curvature, right_curvature, dist_from_center, plot=False):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    # Draw text on image
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    text_color = (255, 255, 255)
    thickness = 2
    result = cv2.putText(result, "Left lane radius of curvature: %(left_curvature).2f meters" % locals(), (100, 100), font, font_scale, text_color, thickness)
    result = cv2.putText(result, "Right lane radius of curvature: %(right_curvature).2f meters" % locals(), (100, 150), font, font_scale, text_color, thickness)
    result = cv2.putText(result, "Vehicle is %(dist_from_center).2f meters of center" % locals(), (100, 200), font, font_scale, text_color, thickness)

    if (plot):
        plt.imshow(result)
        plt.show()

    return result

# Global vars to represent the detected lines in the lane.
left_line = Line()
right_line = Line()

def processImage(image, plot=False):
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
    # visualizeImages(image, combined_grad_binary_x_with_hls_binary, 'Sobel X gradient with S channel', True)

    # Warp the image.
    warped_image, src_points, dst_points, M, Minv = warpImage(combined_grad_binary_x_with_hls_binary)
    # image = drawLines(image, src_points)
    # visualizeImages(image, warped_image, 'Warped image', True)
    if (left_line.detected and right_line.detected):
        # Calculate from previously calculated lines
        left_line_lane_inds, right_line_lane_inds, left_line_current_fit, right_line_current_fit = fitPolynomialAroundLinePositions(warped_image, left_line, right_line)
        # Check if we should keep or drop the frame.
        left_curvature, right_curvature, dist_from_center = calculateRadiusOfCurvatureAndCenterInWorldSpace(image, warped_image, left_line_lane_inds, right_line_lane_inds)
        _, leftx, _, rightx, _ = getNonzeroPositions(warped_image, left_line_lane_inds, right_line_lane_inds)
        if (abs(left_curvature - right_curvature) <= Line.curvature_threshold and
                abs(leftx[-1] - rightx[-1]) <= Line.horizontal_distance_threshold):
            left_line.curvature = left_curvature
            left_line.lane_inds = left_line_lane_inds
            left_line.current_fit = left_line_current_fit
            left_line.allx = leftx
            right_line.curvature = right_curvature
            right_line.lane_inds = right_line_lane_inds
            right_line.current_fit = right_line_current_fit
            right_line.allx = rightx
            left_line.averageCurrentFitWithBestFit()
            right_line.averageCurrentFitWithBestFit()
            Line.dropped_frame_count = 0
        else:
            Line.dropped_frame_count += 1
            print("Dropped Frames: ", Line.dropped_frame_count)
            print("Lane dists: ", abs(leftx[-1] - rightx[-1]))
            print("Curvature diff : ", abs(left_curvature - right_curvature))
            # Too Many frames have been dropped consecutively. Force window recalculation.
            if (Line.dropped_frame_count >= 10):
                left_line.detected = False
                right_line.detected = False
                Line.dropped_frame_count = 0
    else:
        # Calculate from scratch.
        left_line.lane_inds, right_line.lane_inds, left_line.current_fit, right_line.current_fit = slideWindowsFitPolynomial(warped_image)
        _, left_line.allx, _, right_line.allx, _ = getNonzeroPositions(warped_image, left_line.lane_inds, right_line.lane_inds)
        left_line.curvature, right_line.curvature, dist_from_center = calculateRadiusOfCurvatureAndCenterInWorldSpace(image, warped_image, left_line.lane_inds, right_line.lane_inds)
        if (abs(left_line.curvature - right_line.curvature) <= Line.curvature_threshold and
                abs(left_line.allx[-1] - right_line.allx[-1]) <= Line.horizontal_distance_threshold):
            left_line.averageCurrentFitWithBestFit()
            right_line.averageCurrentFitWithBestFit()
            left_line.detected = True
            right_line.detected = True
    return drawLane(image, warped_image, left_line.best_fit, right_line.best_fit, Minv, left_line.curvature, right_line.curvature, dist_from_center, plot)


def main():
    args = defineFlags()
    if (args.force_calibration or not os.path.isfile(CONST_CAMERA_CALIB_PICKLE_FILE_PATH)):
        objpoints, imgpoints = getObjectAndImagePoints()
        prepareCamera(objpoints, imgpoints)
    if (args.video):
        output_video = 'project_video_output.mp4'
        clip = VideoFileClip('project_video.mp4')
        output_clip = clip.fl_image(processImage) #NOTE: this function expects color images!!
        output_clip.write_videofile(output_video, audio=False)
    else:
        # image = undistortImage(mpimg.imread('test_images/straight_lines1.jpg'))
        image = mpimg.imread('test_images/test5.jpg')
        processImage(image, plot=True)

main()
