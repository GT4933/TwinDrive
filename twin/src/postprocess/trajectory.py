import numpy as np
import cv2


class TrajectoryEngine:
    """
    Processes raw binary segmentation masks into physical telemetry,
    trajectory polynomials, cleaned HUD masks, and mode-switching logic.
    """
    def __init__(self, image_shape=(640, 360)):
        self.W, self.H = image_shape[0], image_shape[1]
        self.cam_center_x = self.W // 2
        self.cm_per_pixel = 1.5
        # Half-width of a single lane at the bottom evaluation row in a 640x360 frame
        self.single_lane_half_width_px = 210

    def process_frame(self, drivable_mask, lane_mask):
        """Extracts lane boundaries, cleans masks, and computes telemetry offsets."""
        traj_data = {
            "mode": "UNMARKED_CORRIDOR_FALLBACK",
            "left_x": None, "left_y": None,
            "right_x": None, "right_y": None,
            "lane_center_x": self.cam_center_x,
            "offset_cm": 0.0,
            "direction": "CENTER",
            "clean_drivable": None,
            "clean_lanes": None,
        }

        # 1. ROI Mask: Ignore upper 45% of the image (sky, trees, building facades)
        roi_mask = np.zeros_like(lane_mask)
        roi_mask[int(self.H * 0.45):, :] = 1

        raw_lanes = cv2.bitwise_and(lane_mask, lane_mask, mask=roi_mask)
        raw_drivable = cv2.bitwise_and(drivable_mask, drivable_mask, mask=roi_mask)

        # 2. Isolate the largest GROUNDED drivable blob (Hood-aware up to 36% dashboard height)
        clean_drivable = np.zeros_like(raw_drivable)
        num_d, d_labels, d_stats, d_centroids = cv2.connectedComponentsWithStats(raw_drivable, connectivity=8)
        if num_d > 1:
            # Sort candidate blobs by area descending
            sorted_indices = 1 + np.argsort(d_stats[1:, cv2.CC_STAT_AREA])[::-1]
            for idx in sorted_indices:
                area = d_stats[idx, cv2.CC_STAT_AREA]
                bottom_y = d_stats[idx, cv2.CC_STAT_TOP] + d_stats[idx, cv2.CC_STAT_HEIGHT]
                cx = d_centroids[idx][0]

                # Must cover >= 1.5% of frame, reach below y = 0.64*H (allows tall dashboards/wipers),
                # and have its center of mass within the middle 56% of the camera view
                if (
                    area >= int(self.H * self.W * 0.015)
                    and bottom_y >= int(self.H * 0.64)
                    and (self.W * 0.22 <= cx <= self.W * 0.78)
                ):
                    clean_drivable = (d_labels == idx).astype(np.uint8)
                    break

        traj_data["clean_drivable"] = clean_drivable

        # Create a dilated road zone to verify that candidate lanes sit near actual road surface
        has_drivable = np.count_nonzero(clean_drivable) > 0
        if has_drivable:
            road_zone = cv2.dilate(
                clean_drivable, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 25))
            )
            drivable_ys = np.where(clean_drivable > 0)[0]
            drivable_top_y = int(np.min(drivable_ys))
        else:
            road_zone = None
            drivable_top_y = int(self.H * 0.55)

        # 3. Clean Lane Mask: Sever horizontal crosswalks & remove tiny specks BEFORE closing
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 3))
        horiz_bars = cv2.morphologyEx(raw_lanes, cv2.MORPH_OPEN, horiz_kernel)
        horiz_bars = cv2.dilate(horiz_bars, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)))
        severed_lanes = cv2.subtract(raw_lanes, horiz_bars)

        num_l, l_labels, l_stats, _ = cv2.connectedComponentsWithStats(severed_lanes, connectivity=8)
        filtered_lanes = np.zeros_like(severed_lanes)
        for i in range(1, num_l):
            if l_stats[i, cv2.CC_STAT_AREA] >= 45:
                filtered_lanes[l_labels == i] = 1

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 19))
        clean_lanes = cv2.morphologyEx(filtered_lanes, cv2.MORPH_CLOSE, close_kernel)

        # 4. Contour Extraction & Geometric Filtering
        contours, _ = cv2.findContours(clean_lanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_lines = []
        valid_lane_mask = np.zeros_like(clean_lanes)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Rule 1: Minimum vertical span
            if h < 40:
                continue

            # Rule 2: Ribbon width check
            rect = cv2.minAreaRect(cnt)
            length = max(rect[1][0], rect[1][1], 1.0)
            ribbon_width = area / length
            if ribbon_width > 42.0 or length < 45.0:
                continue

            # Rule 3: Grounding (Must reach lower 38% of screen unless h >= 75)
            bottom_y = np.max(cnt[:, 0, 1])
            if h < 75 and bottom_y < int(self.H * 0.62):
                continue

            # Rule 4: Road Proximity Gate (Rejects glare on vans/obstacles when road is blocked)
            pts = cnt.reshape(-1, 2)
            if road_zone is not None:
                overlap = np.mean(road_zone[pts[:, 1], pts[:, 0]] > 0)
                if overlap < 0.20:
                    continue
            else:
                # If no drivable road is visible at all, require a long, unmistakable lane line
                if h < 75 or area < 280:
                    continue

            valid_lines.append(pts)
            cv2.drawContours(valid_lane_mask, [cnt], -1, 1, thickness=cv2.FILLED)

        traj_data["clean_lanes"] = valid_lane_mask

        if not valid_lines:
            return self._fallback_to_drivable(clean_drivable, traj_data)

        # 5. Split Lane Candidates into Left and Right relative to the BOTTOM of the contour
        left_lines, right_lines = [], []
        for line in valid_lines:
            bottom_x = line[np.argmax(line[:, 1]), 0]
            if bottom_x < self.cam_center_x:
                left_lines.append(line)
            else:
                right_lines.append(line)

        left_pts = [max(left_lines, key=len)] if left_lines else []
        right_pts = [max(right_lines, key=len)] if right_lines else []

        # 6. Dynamic Horizon & Polynomial Curve Fitting
        all_valid_ys = np.concatenate([pts[:, 1] for pts in valid_lines])
        top_y = max(int(self.H * 0.55), min(int(np.min(all_valid_ys)), drivable_top_y + 15))
        plot_y = np.linspace(top_y, self.H - 1, num=20).astype(np.int32)

        left_fit_x = self._fit_curve(left_pts, plot_y, side="LEFT")
        right_fit_x = self._fit_curve(right_pts, plot_y, side="RIGHT")

        # Prevent left/right crossover
        if left_fit_x is not None and right_fit_x is not None:
            if left_fit_x[0] > right_fit_x[0] + 20 or left_fit_x[-1] >= right_fit_x[-1] - 30:
                if len(left_pts[0]) >= len(right_pts[0]):
                    right_fit_x = None
                else:
                    left_fit_x = None

        # 7. Arbitration & Telemetry Calculation
        if left_fit_x is not None or right_fit_x is not None:
            traj_data["mode"] = "LANE_GUIDANCE"

            if left_fit_x is not None:
                traj_data["left_x"], traj_data["left_y"] = left_fit_x, plot_y
            if right_fit_x is not None:
                traj_data["right_x"], traj_data["right_y"] = right_fit_x, plot_y

            # Evaluate lateral center at a lookahead row just above typical hood height (index ~14 -> ~0.85*H)
            eval_idx = -5
            eval_y = plot_y[eval_idx]

            if left_fit_x is not None and right_fit_x is not None:
                center_x = (left_fit_x[eval_idx] + right_fit_x[eval_idx]) / 2.0
            elif left_fit_x is not None:
                # Try fusing with the right edge of clean_drivable if available
                right_edge_x = self._get_drivable_edge(clean_drivable, eval_y, side="RIGHT")
                if right_edge_x is not None and right_edge_x > left_fit_x[eval_idx] + 120:
                    center_x = (left_fit_x[eval_idx] + right_edge_x) / 2.0
                else:
                    center_x = left_fit_x[eval_idx] + self.single_lane_half_width_px
            else:
                # Try fusing with the left edge of clean_drivable if available
                left_edge_x = self._get_drivable_edge(clean_drivable, eval_y, side="LEFT")
                if left_edge_x is not None and left_edge_x < right_fit_x[eval_idx] - 120:
                    center_x = (left_edge_x + right_fit_x[eval_idx]) / 2.0
                else:
                    center_x = right_fit_x[eval_idx] - self.single_lane_half_width_px

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

    def _get_drivable_edge(self, clean_drivable, target_y, side="RIGHT"):
        """Finds the outer x-coordinate of the drivable mask near target_y."""
        if np.count_nonzero(clean_drivable) == 0:
            return None
        y_min = max(0, target_y - 15)
        y_max = min(self.H, target_y + 15)
        band = clean_drivable[y_min:y_max, :]
        xs = np.where(band > 0)[1]
        if len(xs) < 30:
            return None
        return int(np.percentile(xs, 95)) if side == "RIGHT" else int(np.percentile(xs, 5))

    def _fit_curve(self, pts_list, plot_y, degree=2, side=None):
        """Fits a polynomial curve with skeletonization and perspective-aware slope bounds."""
        if not pts_list:
            return None

        all_pts = np.vstack(pts_list)

        unique_ys = np.unique(all_pts[:, 1])
        median_xs = np.array([np.median(all_pts[all_pts[:, 1] == y, 0]) for y in unique_ys])

        xs, ys = median_xs, unique_ys
        y_span = np.max(ys) - np.min(ys)
        bottom_y_pts = np.max(ys)

        if len(ys) < 6 or y_span < 30:
            return None

        linear_slope, _ = np.polyfit(ys, xs, 1)

        if side is not None:
            # Allow wide-angle curb slopes up to 3.6, but enforce perspective direction:
            # Left lanes must angle left as y increases (slope < -0.12)
            # Right lanes must angle right as y increases (slope > +0.12)
            if abs(linear_slope) > 3.60:
                return None
            if side == "LEFT" and linear_slope > -0.12:
                return None
            if side == "RIGHT" and linear_slope < 0.12:
                return None

        if degree == 2 and (y_span < 100 or bottom_y_pts < self.H * 0.80):
            degree = 1

        try:
            coeffs = np.polyfit(ys, xs, degree)
            if degree == 2:
                fit_x = coeffs[0] * (plot_y ** 2) + coeffs[1] * plot_y + coeffs[2]
            else:
                fit_x = coeffs[0] * plot_y + coeffs[1]

            if side == "RIGHT" and fit_x[0] < int(self.W * 0.20):
                return None
            if side == "LEFT" and fit_x[0] > int(self.W * 0.80):
                return None

            return fit_x.astype(np.int32)
        except (np.linalg.LinAlgError, TypeError, np.RankWarning):
            return None

    def _fallback_to_drivable(self, clean_drivable, traj_data):
        """If lanes are missing, calculates center-of-mass and extracts outer corridor edges."""
        if np.count_nonzero(clean_drivable) == 0:
            return traj_data

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

        left_pts, right_pts = [], []
        drivable_ys = np.where(clean_drivable > 0)[0]
        start_y = max(int(self.H * 0.55), int(np.min(drivable_ys)))

        for y in range(self.H - 1, start_y, -5):
            row = clean_drivable[y, :]
            x_indices = np.where(row > 0)[0]

            if len(x_indices) > 25:
                left_pts.append([x_indices[0], y])
                right_pts.append([x_indices[-1], y])

        plot_y = np.linspace(start_y, self.H - 1, num=20).astype(np.int32)

        if len(left_pts) >= 4:
            traj_data["left_x"] = self._fit_curve([np.array(left_pts)], plot_y, degree=1, side=None)
            if traj_data["left_x"] is not None:
                traj_data["left_y"] = plot_y

        if len(right_pts) >= 4:
            traj_data["right_x"] = self._fit_curve([np.array(right_pts)], plot_y, degree=1, side=None)
            if traj_data["right_x"] is not None:
                traj_data["right_y"] = plot_y

        return traj_data