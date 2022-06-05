import cv2
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from scipy.spatial import distance_matrix

import time

DISPLAY = False

SAMPLE_SIZE = 150
DBSCAN_EPSILON = 0.7
HALF_LANE_SHIFT = 125
CANNY_THRESH_LOW = 550
CANNY_THRESH_HIGH = 650
EXIT_COVERAGE_PERCENT = 0.7
MAX_CONTOUR_AREA = 250
MIN_CONTOUR_AREA = 30
PRESPECTIVE_BGCOLOR = (90, 90, 90)
CROP_START_Y = 20
CROP_END_Y = 120
CROP_START_X = 40
CROP_END_X = 40

def prepare_for_canny(img):
    img = cv2.cvtColor(cv2.resize(img, None, fx=0.2, fy=0.2), cv2.COLOR_BGR2GRAY)
    img[np.where(img >= 150)] = 150
    return img

def get_lane_points(img):
    # Resize image to deal with less data
    img = prepare_for_canny(img)
    # Find lane edges
    edges = cv2.Canny(img, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    if DISPLAY:
        cv2.imshow("edges", cv2.resize(edges, None, fx=5, fy=5))
    kernel = np.ones([6,1])
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # cv2.imshow("edges_dilation", cv2.resize(edges, None, fx=2, fy=2))
    # edges = cv2.erode(edges, kernel, iterations=8)

    # Extract sample points and rescale
    points = np.argwhere(edges[:,:] > 128)
    if points.shape[0] == 0:
        return points
    points = points[np.random.choice(
        len(points), size=min(SAMPLE_SIZE,len(points)-1), replace=False
    )]
    points = points * 5
    return points

def find_clusters(points):
    # Normalize the coordinates of the points
    x = StandardScaler().fit_transform(points)

    # DBSCAN model to cluster the points
    model = DBSCAN(eps=DBSCAN_EPSILON)
    
    return model.fit(x).labels_

def get_lane_curves(points, labels, img):
    ordered_labels = get_ordered_labels(points, labels)
    cluster_count = len(set(ordered_labels)) - (1 if -1 in ordered_labels else 0)
    # Distribute the points into separate arrays
    lane_lines = [points[np.where(ordered_labels == i)] for i in range(cluster_count)]
    
    return filter_curves(lane_lines, img)

def get_ordered_labels(data, labels):
    # Get the number of clusters excluding the noise
    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

    # Calculate centre-point of each cluster
    centres = np.array([np.sum(data[np.where(labels == i), 1]) / np.sum(np.where(labels == i)) for i in range(cluster_count)])
    
    # Sort centres ascendingly and get mapping from old labels to new labels
    label_to_index = np.argsort(np.argsort(centres))

    # return labels renumbered (ignore -1 labels)
    labels[labels != -1] = label_to_index[labels[labels != -1]]
    return labels

def get_coverage(params, points, shape, n_samples=20, max_distance=None):
    if max_distance is None:
        # Default max distance 5% of image height
        max_distance = shape[0] / 20
    
    # Get the upper vertex of the line
    # If upper vertex intersects upper border
    if int(params[1]) in range(shape[1]):
        upper_x = 0
    # If upper vertex intersects left border
    elif params[1] < 0:
        upper_x = -params[1] / params[0]
    # If upper vertex intersects right border
    else:
        upper_x = (shape[1] - params[1]) / params[0]
    
    # Get the lower vertex of the line
    # If lower vertex intersects lower border
    if int(np.polyval(params, shape[0])) in range(shape[1]):
        lower_x = shape[0]
    # If lower vertex intersects left border
    elif params[0] > 0:
        lower_x = (shape[1] - params[1]) / params[0]
    # If lower vertex intersects right border
    else:
        lower_x = -params[1] / params[0]
    
    # Create sample points from the given line
    samples = np.linspace(upper_x, lower_x, n_samples)
    test_points = np.column_stack((samples, np.polyval(params, samples)))

    # Calculate distances from the sample points to the given list of points
    dmat = distance_matrix(test_points, points)
    # Calculate coverage percentage by counting sample points that are
    # near to at least one point from the given list of points
    return np.sum(np.min(dmat, axis=1) < max_distance) / dmat.shape[0]


def filter_curves(lines, img):
    curves = []
    for line in lines:
        # fit a straight line to the points
        params = np.polyfit(line[:, 0], line[:, 1], 1)
        curves.append(params)
    # If no curves detected ignore this frame
    if len(curves) == 0:
        return dict()

    # Get the index of the middle lane line
    mid_lane_index = get_midlane_index(curves, img)
    
    # If not found ignore this frame
    if mid_lane_index == -1:
        return dict()

    exit_curve = None
    for i in range(len(curves)):
        # Don't compare middle line to itself
        if i == mid_lane_index:
            continue
        # Calculate the angle between the curve and middle curve
        slope = curves[i][0]
        mid_slope = curves[mid_lane_index][0]
        angle = np.math.atan2(slope - mid_slope, 1 + slope * mid_slope)
        # Lines to the right of the middle curve have the opposite angle
        shift = HALF_LANE_SHIFT
        if i > mid_lane_index:
            angle = -angle
            shift = -shift
        # Check for the correct angle range and coverage
        if angle > np.math.pi / 6 and \
            angle < np.math.pi / 2.5 and \
                get_coverage(curves[i], lines[i], img.shape) >= EXIT_COVERAGE_PERCENT:
            exit_curve = shift_line(curves[i], shift)
            break
    
    # Generate the lane curves by shifting the middle line
    curves = {
        "left": shift_line(curves[mid_lane_index], -HALF_LANE_SHIFT),
        "right": shift_line(curves[mid_lane_index], HALF_LANE_SHIFT)
    }
    
    # If there is an exit curve add it at the end
    if exit_curve is not None:
        curves["exit"] = exit_curve
    return curves

def get_midlane_index(curves, img):
    # Get centres of rectangular contours
    midlane_points = get_midlane_points(img)

    # Count the number of points near to each curve
    curve_point_count = np.zeros((len(curves),))
    for i, curve in enumerate(curves):
        for point in midlane_points:
            distance = abs(point[0] - curve[0] * point[1] - curve[1]) / np.sqrt(1 + curve[0] ** 2)
            if distance / img.shape[0] < 0.1:
                curve_point_count[i] += 1
    
    # Get the index of the curve with maximum number of points
    try:
        max_index = int(np.argmax(curve_point_count))
    except:
        max_index = -1

    # if max_index != -1 and curve_point_count[max_index] > 0:
    #     draw_curve(img, curves[max_index])
    # cv2.imshow("img1", cv2.resize(img, None, fx=0.5, fy=0.5))

    # Return the index if it has points near to it otherwise return -1 (not found)
    return max_index if max_index != -1 and curve_point_count[max_index] > 0 else -1

def get_midlane_points(img):
    # Edge detection to find lane lines
    img = prepare_for_canny(img)
    edges = cv2.Canny(img , CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    kernel = np.ones([3,3])
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Finding all contours in the edge image
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    points = []
    for i in contours:
        # Filtering Contours by area
        area = cv2.contourArea(i)
        if area > MAX_CONTOUR_AREA or area < MIN_CONTOUR_AREA:
            continue

        # Approximating number of vertices for contours
        epsilon = 0.05*cv2.arcLength(i,True)
        approx = cv2.convexHull(i)
        approx = cv2.approxPolyDP(approx,epsilon,True)

        # Filtering Contours by number of vertices
        if len(approx) >= 4:
            M = cv2.moments(approx)
            points.append((int(M['m10']/M['m00']) * 5, int(M['m01']/ M['m00']) * 5))
    
    # img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, fx=5, fy=5)
    # for point in points:
    #     cv2.circle(img, point, 10, (0,0,255), -1)
    # cv2.imshow("img2", cv2.resize(img, None, fx=1, fy=1))
    # cv2.imshow("edges", cv2.resize(edges, None, fx=5, fy=5))

    return points

def shift_line(line, shift):
    new_curve = np.copy(line)
    new_curve[1] = shift * np.sqrt(new_curve[0] ** 2 + 1) + new_curve[1]
    return new_curve

def draw_curve(img, curve, color=(0,0,255), thickness=3, point_curve=False):
    if point_curve:
        # If points are provided directly
        x = curve[:, 0].astype(np.int)
        y = curve[:, 1].astype(np.int)
    else:
        # If parameters are provided calculate the points
        x = np.array([i for i in range(0, img.shape[0], 10)])
        y = np.polyval(curve, x).astype(np.int)
        cv2.putText(img, f"{curve[0]: .3f}", (np.polyval(curve, img.shape[1] // 2).astype(np.int), img.shape[1] // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

    for i in range(1, len(x)):
        cv2.line(img, (y[i], x[i]), (y[i - 1], x[i - 1]), color=color, thickness=thickness)
    return img

def draw_lane(img, curve1, curve2, color, opacity=0.6):
    x = np.array([i for i in range(0, img.shape[0], 10)], dtype=np.int)
    y1 = np.polyval(curve1, x).astype(np.int)
    y2 = np.polyval(curve2, x).astype(np.int)
    points = np.row_stack((np.column_stack((y1, x)), np.column_stack((y2, x))[::-1]))
    overlay = img.copy()
    cv2.fillPoly(overlay, [points], color=color)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

def draw_lane_from_curve(img, curve, shift, color):
    curve1 = shift_line(curve, -shift)
    curve2 = shift_line(curve, shift)
    draw_lane(img, curve1, curve2, color)

def render_clusters(points, labels, img_shape):
    img = np.zeros(img_shape)
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    for i, point in enumerate(points):
        if labels[i] == -1:
            continue
        cv2.circle(img, (int(point[1]), int(point[0])), 3, colors[labels[i] % len(colors)], -1)

    return img

def prespective_transform(img, inverse=False, border_value=PRESPECTIVE_BGCOLOR):
    presp_pts1 = np.float32([[0, img.shape[0] // 2], [img.shape[1], img.shape[0] // 2],
                    [img.shape[1] // 12, img.shape[0] // 4], [11 * img.shape[1] // 12, img.shape[0] // 4]])
    
    presp_pts2 = np.float32([[0, img.shape[0] // 2], [img.shape[1], img.shape[0] // 2],
                    [0, 0], [img.shape[1], 0]])
    
    # result = img
    # for pt in presp_pts1:
    #     cv2.circle(result, (pt[0], pt[1]), 5, (0, 255, 255), thickness=-1)
    # cv2.fillPoly(img, [np.array([presp_pts1[0], presp_pts1[1], presp_pts1[3], presp_pts1[2]], dtype=int)], color=(0,255,255))

    if inverse:
        presp_mat = cv2.getPerspectiveTransform(presp_pts2, presp_pts1)
    else:
        presp_mat = cv2.getPerspectiveTransform(presp_pts1, presp_pts2)

    # Apply prespective transform and fill background with grey pixels
    result = cv2.warpPerspective(img, presp_mat, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return result

def get_curve_value(curves, key, img_shape):
    if key not in curves:
        return None

    curve = curves[key]
    curve_midpoint = np.polyval(curve, img_shape[0] // 2)
    return min(1, max(-1 ,(curve_midpoint - img_shape[1] // 2) / (img_shape[1] // 2)))

def detect_lanes(img):
    img = cv2.resize(img[CROP_START_Y:img.shape[0] - CROP_END_Y,CROP_START_X:img.shape[1]-CROP_END_X,:], None, fx=1, fy=1)
    
    # img_presp = prespective_transform(img)
    img_presp = img.copy()

    points = get_lane_points(img_presp)
    if points.shape[0] == 0:
        return {
            "image": img,
            "image_prespective": img_presp,
            "left_lane": None,
            "right_lane": None,
            "exit_lane": None
        }

    labels = find_clusters(points)

    curves = get_lane_curves(points, labels, img_presp)

    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    
    for i, (name, curve) in enumerate(curves.items()):
        draw_lane_from_curve(img_presp, curve, HALF_LANE_SHIFT, colors[i])
        val = get_curve_value(curves, name, img.shape)
        cv2.putText(img_presp, str(val), (np.polyval(curve, img_presp.shape[0] - 20).astype(int), img_presp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[3-i], 2)
    
    # img_presp = render_clusters(points, labels, img.shape)
    
    return {
        "image": img,
        "image_prespective": img_presp,
        "left_lane": get_curve_value(curves, "left", img_presp.shape),
        "right_lane": get_curve_value(curves, "right", img_presp.shape),
        "exit_lane": get_curve_value(curves, "exit", img_presp.shape)
    }

def main():
    cap = cv2.VideoCapture("video001.avi")
    last_time = time.time()

    paused = False
    while True:
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_number == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()
        
        detection = detect_lanes(frame)
        img = detection["image"]
        img_presp = detection["image_prespective"]

        if DISPLAY:
            cv2.imshow("img", img)
            cv2.imshow("imgpresp", img_presp)

        if paused:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = True
        elif key == ord('r'):
            paused = False
        elif key == ord('b') and paused:
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 2)
        elif key == ord('s') and paused:
            cv2.imwrite(f"frame{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg", frame)
        
        FPS = 1 / (time.time() - last_time)
        last_time = time.time()
        print("{:.2f} FPS".format(FPS))

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
