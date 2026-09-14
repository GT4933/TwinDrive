import numpy as np
import cv2

class TrajectoryEngine:
    """
    Processes raw binary segmentation masks into physical telemetry,
    trajectory polynomials, and mode-switching logic.
    """
    def __init__(self, image_shape=(640, 360)):
        self.W, self.H = image_shape[0], image_shape[1]
        self.cam_center_x = self.W // 2
        
        # Rough physical calibration: Assuming a standard lane is ~3.7 meters wide
        # and occupies ~250 pixels at the bottom edge of a 640x360 frame.
        # 370 cm / 250 px = ~1.5 cm/px
        self.cm_per_pixel = 1.5 

    def process_frame(self, drivable_mask, lane_mask):
        """Extracts lane boundaries and computes telemetry offsets."""
        
        # Default fallback payload
        traj_data = {
            "mode": "UNMARKED_CORRIDOR_FALLBACK",
            "left_x": None, "left_y": None,
            "right_x": None, "right_y": None,
            "lane_center_x": self.cam_center_x,
            "offset_cm": 0.0,
            "direction": "CENTER"
        }
        
        # 1. ROI Mask: Ignore upper 45% of the image (sky, trees, powerlines)
        roi_mask = np.zeros_like(lane_mask)
        roi_mask[int(self.H * 0.45):, :] = 1
        clean_lanes = cv2.bitwise_and(lane_mask, lane_mask, mask=roi_mask)

        # A tall, narrow kernel (3x35) connects vertical dashes without bridging horizontal noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 35))
        clean_lanes = cv2.morphologyEx(clean_lanes, cv2.MORPH_CLOSE, kernel)
        
        # 2. Area Thresholding (Must be at least 0.4% of the image to be a real lane)
        min_pixels = int(self.H * self.W * 0.004)
        if np.count_nonzero(clean_lanes) < min_pixels:
            return self._fallback_to_drivable(drivable_mask, traj_data)
            
        # 3. Contour Extraction & Filtering
        contours, _ = cv2.findContours(clean_lanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_lines = []
        for cnt in contours:
            # Reject tiny noise artifacts
            if cv2.contourArea(cnt) < 100:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            # Reject horizontal lines (bumpers, crosswalks) by requiring vertical height
            if h < 30: 
                continue
                
            valid_lines.append(cnt.reshape(-1, 2))
            
        if not valid_lines:
            return self._fallback_to_drivable(drivable_mask, traj_data)
            
        # 4. Split Lane Candidates into Left and Right relative to camera center
        left_pts, right_pts = [], []
        for line in valid_lines:
            if np.mean(line[:, 0]) < self.cam_center_x:
                left_pts.append(line)
            else:
                right_pts.append(line)
                
        # 5. Polynomial Curve Fitting (x as a function of y)
        # Create y-coordinates starting from the mid-horizon to the bottom of the frame
        plot_y = np.linspace(int(self.H * 0.55), self.H - 1, num=20).astype(np.int32)
        
        left_fit_x = self._fit_curve(left_pts, plot_y)
        right_fit_x = self._fit_curve(right_pts, plot_y)
        
        # 6. Arbitration & Telemetry Calculation
        if left_fit_x is not None or right_fit_x is not None:
            traj_data["mode"] = "LANE_GUIDANCE"
            
            if left_fit_x is not None:
                traj_data["left_x"], traj_data["left_y"] = left_fit_x, plot_y
            if right_fit_x is not None:
                traj_data["right_x"], traj_data["right_y"] = right_fit_x, plot_y
                
            # Calculate the target lane center specifically at the bottom edge (closest to car)
            bottom_idx = -1 
            if left_fit_x is not None and right_fit_x is not None:
                center_x = (left_fit_x[bottom_idx] + right_fit_x[bottom_idx]) / 2.0
            elif left_fit_x is not None:
                # If only left lane is visible, project center ~125px to the right
                center_x = left_fit_x[bottom_idx] + 125 
            else:
                # If only right lane is visible, project center ~125px to the left
                center_x = right_fit_x[bottom_idx] - 125
                
            # Compute lateral offset metrics
            # 1. Clamp the raw center_x to prevent wild off-screen extrapolations
            center_x = np.clip(center_x, -100, self.W + 100)
            traj_data["lane_center_x"] = np.clip(center_x, 0, self.W)
            
            offset_px = center_x - self.cam_center_x
            
            # 2. Calculate cm and strictly cap the maximum logical offset (~4.5 meters)
            raw_offset_cm = abs(offset_px * self.cm_per_pixel)
            traj_data["offset_cm"] = float(np.clip(raw_offset_cm, 0.0, 450.0))
            
            # Deadzone of 10 pixels before declaring a side
            if offset_px > 10:
                traj_data["direction"] = "RIGHT"
            elif offset_px < -10:
                traj_data["direction"] = "LEFT"
            else:
                traj_data["direction"] = "CENTER"
        else:
            return self._fallback_to_drivable(drivable_mask, traj_data)
            
        return traj_data

    def _fit_curve(self, pts_list, plot_y):
        """Fits a degree-2 polynomial x = a*y^2 + b*y + c to avoid vertical slope singularities."""
        if not pts_list:
            return None
            
        all_pts = np.vstack(pts_list)
        xs, ys = all_pts[:, 0], all_pts[:, 1]
        
        try:
            coeffs = np.polyfit(ys, xs, 2)
            fit_x = coeffs[0] * (plot_y ** 2) + coeffs[1] * plot_y + coeffs[2]
            return fit_x.astype(np.int32)
        except np.linalg.LinAlgError:
            return None
            
    def _fallback_to_drivable(self, drivable_mask, traj_data):
        """If lanes are missing, calculates center-of-mass and extracts outer corridor edges."""
        # 1. Calculate Telemetry (Center of Mass)
        M = cv2.moments(drivable_mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            traj_data["lane_center_x"] = cx
            offset_px = cx - self.cam_center_x
            
            raw_offset_cm = abs(offset_px * self.cm_per_pixel)
            traj_data["offset_cm"] = float(np.clip(raw_offset_cm, 0.0, 450.0))
            
            if offset_px > 10:
                traj_data["direction"] = "RIGHT"
            elif offset_px < -10:
                traj_data["direction"] = "LEFT"
            else:
                traj_data["direction"] = "CENTER"
                
        # 2. Extract Outer Boundaries for HUD Visuals (Blue Curves)
        left_pts = []
        right_pts = []
        start_y = int(self.H * 0.55)
        
        # Scan from the bottom of the screen upwards to find the edges of the green mask
        for y in range(self.H - 1, start_y, -5):
            row = drivable_mask[y, :]
            x_indices = np.where(row > 0)[0]
            
            if len(x_indices) > 10: # Require a minimum width to count as a valid surface
                left_pts.append([x_indices[0], y])      # Left-most green pixel
                right_pts.append([x_indices[-1], y])    # Right-most green pixel
                
        # 3. Fit Polynomials to the extracted edges
        plot_y = np.linspace(int(self.H * 0.55), self.H - 1, num=20).astype(np.int32)
        
        if len(left_pts) >= 5:
            traj_data["left_x"] = self._fit_curve([np.array(left_pts)], plot_y)
            if traj_data["left_x"] is not None:
                traj_data["left_y"] = plot_y
                
        if len(right_pts) >= 5:
            traj_data["right_x"] = self._fit_curve([np.array(right_pts)], plot_y)
            if traj_data["right_x"] is not None:
                traj_data["right_y"] = plot_y
                
        return traj_data