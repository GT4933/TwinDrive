import numpy as np
import cv2

class TrajectoryEngine:
    """
    Processes raw binary segmentation masks into physical telemetry,
    trajectory polynomials, and mode-switching logic.
    """
    def __init__(self, image_shape=(640, 360), ema_alpha=0.0):
        self.W, self.H = image_shape[0], image_shape[1]
        self.cam_center_x = self.W // 2

        # Rough physical calibration: Assuming a standard lane is ~3.7 meters wide
        # and occupies ~250 pixels at the bottom edge of a 640x360 frame.
        # 370 cm / 250 px = ~1.5 cm/px
        self.cm_per_pixel = 1.5

        # Temporal smoothing for video: weight of the newest frame in an exponential moving average.
        # 0 disables smoothing (independent images); ~0.3-0.5 suits 30 FPS video.
        self.ema_alpha = ema_alpha
        self.reset()

    def reset(self):
        """Forget smoothing state (call between unrelated images or video clips)."""
        self._ema_center_x = None
        self._ema_fits = {"left_x": None, "right_x": None}

    def process_frame(self, drivable_mask, lane_mask, ego_mask=None):
        """
        Extracts lane boundaries and computes telemetry offsets.
        ego_mask (optional): the model's own-lane prediction, used for a tighter fallback corridor
        when no lane markings are found.
        """

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
            return self._smooth(self._fallback(drivable_mask, ego_mask, traj_data))

        # 3. Contour Extraction & Filtering
        contours, _ = cv2.findContours(clean_lanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_lines = []
        for cnt in contours:
            # Reject tiny noise artifacts
            if cv2.contourArea(cnt) < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Reject horizontal lines (bumpers, crosswalks) by requiring vertical height
            if h < 30:
                continue
            # Reject blobs wider than they are tall (crosswalk patches, painted road text)
            if w > 2 * h:
                continue

            valid_lines.append(cnt.reshape(-1, 2))

        if not valid_lines:
            return self._smooth(self._fallback(drivable_mask, ego_mask, traj_data))

        # 4. Pick the ego-lane boundaries: on each side, the line whose near (bottom) end is closest to
        #    the camera centre. Merging every line on a side would mix in the neighbouring lanes' markings.
        left_line, right_line = self._nearest_lines(valid_lines)

        # 5. Polynomial Curve Fitting (x as a function of y)
        # Create y-coordinates starting from the mid-horizon to the bottom of the frame
        plot_y = np.linspace(int(self.H * 0.55), self.H - 1, num=20).astype(np.int32)

        left_fit_x = self._fit_curve([left_line] if left_line is not None else [], plot_y)
        right_fit_x = self._fit_curve([right_line] if right_line is not None else [], plot_y)

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

            # Clamp the raw center_x to prevent wild off-screen extrapolations
            traj_data["lane_center_x"] = float(np.clip(center_x, -100, self.W + 100))
        else:
            return self._smooth(self._fallback(drivable_mask, ego_mask, traj_data))

        return self._smooth(traj_data)

    def _nearest_lines(self, lines):
        """Returns (left, right): per side, the line whose bottom-most points lie closest to the camera centre."""
        best = {"left": (None, np.inf), "right": (None, np.inf)}
        for line in lines:
            bottom = line[line[:, 1] >= line[:, 1].max() - 10]   # points within 10 px of the line's lowest row
            near_x = float(bottom[:, 0].mean())
            side = "left" if near_x < self.cam_center_x else "right"
            dist = abs(near_x - self.cam_center_x)
            if dist < best[side][1]:
                best[side] = (line, dist)
        return best["left"][0], best["right"][0]

    def _set_offset(self, traj_data):
        """Lateral offset, direction and display clamp from traj_data['lane_center_x']."""
        center_x = traj_data["lane_center_x"]
        offset_px = center_x - self.cam_center_x
        # Calculate cm and strictly cap the maximum logical offset (~4.5 meters)
        traj_data["offset_cm"] = float(np.clip(abs(offset_px * self.cm_per_pixel), 0.0, 450.0))
        # Deadzone of 10 pixels before declaring a side
        if offset_px > 10:
            traj_data["direction"] = "RIGHT"
        elif offset_px < -10:
            traj_data["direction"] = "LEFT"
        else:
            traj_data["direction"] = "CENTER"
        traj_data["lane_center_x"] = float(np.clip(center_x, 0, self.W))
        return traj_data

    def _smooth(self, traj_data):
        """Exponential moving average of the lane centre and fitted curves across video frames."""
        a = self.ema_alpha
        if a > 0:
            if self._ema_center_x is not None:
                traj_data["lane_center_x"] = a * traj_data["lane_center_x"] + (1 - a) * self._ema_center_x
            self._ema_center_x = traj_data["lane_center_x"]
            for key in ("left_x", "right_x"):
                fit = traj_data[key]
                if fit is None:
                    self._ema_fits[key] = None   # side lost: restart its average when it reappears
                    continue
                prev = self._ema_fits[key]
                if prev is not None:
                    fit = (a * fit + (1 - a) * prev).astype(np.int32)
                    traj_data[key] = fit
                self._ema_fits[key] = fit.astype(np.float64)
        return self._set_offset(traj_data)

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

    def _fallback(self, drivable_mask, ego_mask, traj_data):
        """No usable lane markings: follow the predicted ego lane if there is one, else the whole drivable area."""
        min_pixels = int(self.H * self.W * 0.01)
        if ego_mask is not None and np.count_nonzero(ego_mask[int(self.H * 0.55):, :]) >= min_pixels:
            traj_data["mode"] = "EGO_LANE_FALLBACK"
            return self._fallback_to_region(ego_mask, traj_data)
        return self._fallback_to_region(drivable_mask, traj_data)

    def _fallback_to_region(self, region_mask, traj_data):
        """Calculates the region's centre of mass and extracts its outer corridor edges."""
        # 1. Calculate Telemetry (Center of Mass of the near half of the frame)
        near = np.zeros_like(region_mask)
        near[int(self.H * 0.55):, :] = region_mask[int(self.H * 0.55):, :]
        M = cv2.moments(near)
        if M["m00"] > 0:
            traj_data["lane_center_x"] = M["m10"] / M["m00"]

        # 2. Extract Outer Boundaries for HUD Visuals
        left_pts = []
        right_pts = []
        start_y = int(self.H * 0.55)

        # Scan from the bottom of the screen upwards to find the edges of the region
        for y in range(self.H - 1, start_y, -5):
            row = region_mask[y, :]
            x_indices = np.where(row > 0)[0]

            if len(x_indices) > 10: # Require a minimum width to count as a valid surface
                left_pts.append([x_indices[0], y])      # Left-most pixel
                right_pts.append([x_indices[-1], y])    # Right-most pixel

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
