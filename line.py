import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    curvature_threshold = 500
    dropped_frame_count = 0
    horizontal_distance_threshold = 800

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # All good pixel indices
        self.lane_inds = None

    def averageCurrentFitWithBestFit(self):
        if (self.best_fit is None):
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.current_fit + self.best_fit) / 2
