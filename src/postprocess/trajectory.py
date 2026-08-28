import numpy as np
import cv2

class TrajectoryEngine:
    """
    Transforms raw neural network probability masks into geometric vectors,
    quadratic polynomial curves, dynamic vehicle offset metrics, and dynamic fallbacks.
    """
    def __init__(self, image_shape=(640, 360), meter_per_pixel_x=3.7 / 400.0):
        """
        Args:
            image_shape (tuple): Frame resolution (Width, Height) = (640, 360)
            meter_per_pixel_x (float): Horizontal calibration ratio (approx 3.7 meters standard lane width / ~400 pixels)
        """
        self.width, self.height = image_shape
        self.camera_center_x = self.width / 2.0
        self.meter_per_pixel_x = meter_per_pixel_x
        
    def fit_polynomial(self, binary_mask):
        """
        Fits a 2nd-order quadratic polynomial x = a*y^2 + b*y + c to pixel coordinates.
        Returns polynomial coefficients [a, b, c] and plotted x-coordinates.
        """
        y_indices, x_indices = np.where(binary_mask > 0)
        
        if len(y_indices) < 50:  # Not enough active pixels for reliable curve fitting
            return None, None, None

        # Fit quadratic curve (x as function of y to handle vertical/near-vertical lines cleanly)
        poly_coeffs = np.polyfit(y_indices, x_indices, 2)
        
        # Generate target Y evaluation points from horizon to bonnet
        plot_y = np.linspace(int(self.height * 0.55), self.height - 1, num=30)
        plot_x = poly_coeffs[0] * (plot_y ** 2) + poly_coeffs[1] * plot_y + poly_coeffs[2]
        
        return poly_coeffs, plot_x, plot_y

    def extract_corridor_boundaries(self, drivable_mask):
        """
        Fallback logic: When lane markings disappear on dirt or unpaved roads,
        extracts the outer boundary contours of the drivable area mask to synthesize virtual guidelines.
        """
        # Find contours of the drivable space mask
        contours, _ = cv2.findContours(drivable_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None, None

        # Extract largest contour (main road body)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.squeeze()

        if len(contour_points.shape) < 2:
            return None, None, None, None

        # Filter points in lower half of image (bottom perspective near vehicle)
        bottom_pts = contour_points[contour_points[:, 1] >= int(self.height * 0.55)]
        
        if len(bottom_pts) < 20:
            return None, None, None, None

        # Separate left and right edges based on horizontal midpoint
        left_pts = bottom_pts[bottom_pts[:, 0] < self.camera_center_x]
        right_pts = bottom_pts[bottom_pts[:, 0] >= self.camera_center_x]

        left_coeffs, left_x, left_y = self.fit_polynomial_from_pts(left_pts)
        right_coeffs, right_x, right_y = self.fit_polynomial_from_pts(right_pts)

        return left_x, left_y, right_x, right_y

    def fit_polynomial_from_pts(self, pts):
        """Fits poly curve x = a*y^2 + b*y + c directly from array of [x, y] points."""
        if len(pts) < 10:
            return None, None, None
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        
        coeffs = np.polyfit(y_coords, x_coords, 2)
        plot_y = np.linspace(int(self.height * 0.55), self.height - 1, num=30)
        plot_x = coeffs[0] * (plot_y ** 2) + coeffs[1] * plot_y + coeffs[2]
        return coeffs, plot_x, plot_y

    def process_frame(self, drivable_mask, lane_mask):
        """
        Processes multi-head neural outputs and generates navigation telemetry.
        Returns dictionary containing trajectory points, offset metrics, and operation mode.
        """
        lane_pixel_count = np.sum(lane_mask > 0)
        mode = "LANE_GUIDANCE"
        
        left_x, left_y, right_x, right_y = None, None, None, None
        
        # Primary Mode: Dedicated Lane Lines detected
        if lane_pixel_count > 150:
            # Separate left and right lane markings using camera center
            y_pts, x_pts = np.where(lane_mask > 0)
            left_mask = np.zeros_like(lane_mask)
            right_mask = np.zeros_like(lane_mask)
            
            left_indices = x_pts < self.camera_center_x
            right_indices = x_pts >= self.camera_center_x
            
            left_mask[y_pts[left_indices], x_pts[left_indices]] = 1
            right_mask[y_pts[right_indices], x_pts[right_indices]] = 1
            
            _, left_x, left_y = self.fit_polynomial(left_mask)
            _, right_x, right_y = self.fit_polynomial(right_mask)

        # Fallback Mode: Sparse lane activation (Unstructured / Dirt / Unmarked road)
        if left_x is None or right_x is None:
            mode = "UNMARKED_CORRIDOR_FALLBACK"
            left_x, left_y, right_x, right_y = self.extract_corridor_boundaries(drivable_mask)

        # Calculate vehicle offset at bonnet height (y = self.height - 1)
        target_y = self.height - 1
        
        if left_x is not None and right_x is not None:
            lane_center_x = (left_x[-1] + right_x[-1]) / 2.0
        elif left_x is not None:
            lane_center_x = left_x[-1] + 200.0  # Default nominal lane half-width
        elif right_x is not None:
            lane_center_x = right_x[-1] - 200.0
        else:
            # Fallback to drivable space horizontal centroid
            drivable_bottom = drivable_mask[int(self.height * 0.8):, :]
            y_indices, x_indices = np.where(drivable_bottom > 0)
            if len(x_indices) > 0:
                lane_center_x = np.mean(x_indices)
            else:
                lane_center_x = self.camera_center_x

        # Calculate lateral displacement (in cm)
        pixel_offset = self.camera_center_x - lane_center_x
        offset_meters = pixel_offset * self.meter_per_pixel_x
        offset_cm = offset_meters * 100.0

        return {
            "mode": mode,
            "left_x": left_x,
            "left_y": left_y,
            "right_x": right_x,
            "right_y": right_y,
            "lane_center_x": lane_center_x,
            "offset_cm": offset_cm,
            "direction": "RIGHT" if offset_cm < 0 else ("LEFT" if offset_cm > 0 else "CENTER")
        }