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
        
        # 1. ROI Mask: Ignore upper 45% of the image (sky, trees, buildings)
        roi_mask = np.zeros_like(lane_mask)
        roi_mask[int(self.H * 0.45):, :] = 1
        
        # Clean both masks to prevent sky/building hallucinations
        clean_lanes = cv2.bitwise_and(lane_mask, lane_mask, mask=roi_mask)
        clean_drivable = cv2.bitwise_and(drivable_mask, drivable_mask, mask=roi_mask)
        
        # Isolate the largest connected blob in the drivable mask to ignore parked cars/buildings
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_drivable, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean_drivable = (labels == largest_label).astype(np.uint8)
        else:
            clean_drivable = np.zeros_like(clean_drivable)

        # A tall, narrow kernel (3x35) connects vertical dashes without bridging horizontal noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 35))
        clean_lanes = cv2.morphologyEx(clean_lanes, cv2.MORPH_CLOSE, kernel)
        
        # 2. Area Thresholding
        min_pixels = int(self.H * self.W * 0.004)
        if np.count_nonzero(clean_lanes) < min_pixels:
            return self._fallback_to_drivable(clean_drivable, traj_data)
            
        # 3. Contour Extraction & Filtering
        contours, _ = cv2.findContours(clean_lanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_lines = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Rule 1: Vertical Height Requirement
            if h < 40: 
                continue
                
            # Rule 2: True Thickness (Reject massive crosswalk blobs without killing diagonal lanes)
            # minAreaRect finds the rotated width/length, perfectly handling diagonal lines.
            rect = cv2.minAreaRect(cnt)
            box_w, box_h = rect[1]
            thickness = min(box_w, box_h)
            if thickness > 60:  # Real lane lines are thin; crosswalk blocks are fat
                continue
                
            # Rule 3: Grounding (Reject floating raindrops / glare on windshields)
            bottom_y = np.max(cnt[:, 0, 1])
            if h < 80 and bottom_y < self.H * 0.60:
                continue
                
            valid_lines.append(cnt.reshape(-1, 2))
            
        if not valid_lines:
            return self._fallback_to_drivable(clean_drivable, traj_data)
            
        # 4. Split Lane Candidates into Left and Right relative to the BOTTOM of the screen
        left_lines, right_lines = [], []
        for line in valid_lines:
            # Find the x-coordinate of the lowest point on the screen (closest to the car bumper)
            bottom_x = line[np.argmax(line[:, 1]), 0]
            
            if bottom_x < self.cam_center_x:
                left_lines.append(line)
            else:
                right_lines.append(line)
                
        # Isolate the single most dominant line on each side to prevent background noise
        left_pts = [max(left_lines, key=len)] if left_lines else []
        right_pts = [max(right_lines, key=len)] if right_lines else []
                
        # 5. Polynomial Curve Fitting (x as a function of y)
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
                
            bottom_idx = -1 
            if left_fit_x is not None and right_fit_x is not None:
                center_x = (left_fit_x[bottom_idx] + right_fit_x[bottom_idx]) / 2.0
            elif left_fit_x is not None:
                center_x = left_fit_x[bottom_idx] + 125 
            else:
                center_x = right_fit_x[bottom_idx] - 125
                
            center_x = np.clip(center_x, -100, self.W + 100)
            traj_data["lane_center_x"] = np.clip(center_x, 0, self.W)
            
            offset_px = center_x - self.cam_center_x
            raw_offset_cm = abs(offset_px * self.cm_per_pixel)
            traj_data["offset_cm"] = float(np.clip(raw_offset_cm, 0.0, 450.0))
            
            if offset_px > 10:
                traj_data["direction"] = "RIGHT"
            elif offset_px < -10:
                traj_data["direction"] = "LEFT"
            else:
                traj_data["direction"] = "CENTER"
        else:
            return self._fallback_to_drivable(clean_drivable, traj_data)
            
        return traj_data

    def _fit_curve(self, pts_list, plot_y, degree=2):
        """Fits a polynomial curve using a 1-pixel skeleton to prevent thick-blob distortion."""
        if not pts_list:
            return None
            
        all_pts = np.vstack(pts_list)
        
        # SKELETONIZATION
        unique_ys = np.unique(all_pts[:, 1])
        median_xs = np.array([np.median(all_pts[all_pts[:, 1] == y, 0]) for y in unique_ys])
        
        xs, ys = median_xs, unique_ys
        y_span = np.max(ys) - np.min(ys)
        bottom_y_pts = np.max(ys)
        
        if len(ys) < 5 or y_span < 25:
            return None
            
        # SAFETY INTERLOCK (Prevent Parabolic Extrapolation Hooks)
        # If a line is short, OR if it stops high up on the screen and we have to 
        # extrapolate it blindly down to the car's bumper, a degree-2 parabola 
        # will violently hook. Force a straight line instead.
        if degree == 2 and (y_span < 100 or bottom_y_pts < self.H * 0.80):
            degree = 1
            
        try:
            coeffs = np.polyfit(ys, xs, degree)
            if degree == 2:
                fit_x = coeffs[0] * (plot_y ** 2) + coeffs[1] * plot_y + coeffs[2]
            else:
                fit_x = coeffs[0] * plot_y + coeffs[1]
            return fit_x.astype(np.int32)
        except (np.linalg.LinAlgError, TypeError, np.RankWarning):
            return None

    def _fallback_to_drivable(self, clean_drivable, traj_data):
        """If lanes are missing, calculates center-of-mass and extracts outer corridor edges."""
        # 1. Calculate Telemetry (Center of Mass)
        M = cv2.moments(clean_drivable)
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
                
        # 2. Extract Outer Boundaries for HUD Visuals
        left_pts = []
        right_pts = []
        start_y = int(self.H * 0.55)
        
        for y in range(self.H - 1, start_y, -5):
            row = clean_drivable[y, :]
            x_indices = np.where(row > 0)[0]
            
            if len(x_indices) > 10: 
                left_pts.append([x_indices[0], y])      
                right_pts.append([x_indices[-1], y])    
                
        plot_y = np.linspace(int(self.H * 0.55), self.H - 1, num=20).astype(np.int32)
        
        # ALWAYS force degree=1 for the fallback corridors. 
        # Segmented blob edges are too irregular for parabolas.
        if len(left_pts) >= 5:
            traj_data["left_x"] = self._fit_curve([np.array(left_pts)], plot_y, degree=1)
            if traj_data["left_x"] is not None:
                traj_data["left_y"] = plot_y
                
        if len(right_pts) >= 5:
            traj_data["right_x"] = self._fit_curve([np.array(right_pts)], plot_y, degree=1)
            if traj_data["right_x"] is not None:
                traj_data["right_y"] = plot_y
                
        return traj_data