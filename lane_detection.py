import cv2
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import time

SAMPLE_SIZE = 300

def get_lane_points(img):
    # Resize image to deal with less data
    img = cv2.resize(img, None, fx=0.1, fy=0.1)
    # Find lane edges
    edges = cv2.Canny(img, 200, 300)
    # cv2.imshow("edges", edges)
    kernel = np.ones([6,1])
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # cv2.imshow("edges_dilation", edges)
    #edges = cv2.erode(edges, kernel, iterations=8)

    # Extract sample points and rescale
    points = np.argwhere(edges[:,:] > 128)
    points = points[np.random.choice(
        len(points), size=min(SAMPLE_SIZE,len(points)-1), replace=False
    )]
    points = points * 10
    return points

def find_clusters(points):
    # Normalize the coordinates of the points
    x = StandardScaler().fit_transform(points)

    # DBSCAN model to cluster the points
    model = DBSCAN(eps=0.7)
    
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
    centres = np.zeros((cluster_count,), dtype=np.float)
    counts = np.zeros((cluster_count,), dtype=np.float)
    for i, (x, y) in enumerate(data):
        if labels[i] == -1:
            continue
        centres[labels[i]] += y
        counts[labels[i]] += 1

    for i in range(cluster_count):
        centres[i] /= counts[i]
    
    # Sort clusters in ascending order of the center value
    label_to_index = np.zeros((cluster_count,), dtype=np.int)
    indices = np.argwhere(np.ones_like(centres))
    for a, (c, i) in enumerate(sorted(zip(centres, indices))):
        label_to_index[i] = a

    # return labels renumbered
    return np.array([label_to_index[label] if label != -1 else -1 for label in labels])

def filter_curves(lines, img):
    curves = []
    for line in lines:
        # fit a straight line to the points
        params = np.polyfit(line[:, 0], line[:, 1], 1)
        curves.append(params)
    
    # If no curves detected ignore this frame
    if len(curves) == 0:
        return []

    # Get the index of the middle lane line
    mid_lane_index = get_midlane_index(curves, img)
    # If not found ignore this frame
    if mid_lane_index == -1:
        return []
    
    # Remove all curves other than the middle one
    curves = curves[mid_lane_index:mid_lane_index+1]
    mid_lane_index = 0
    
    # Recreate the other lines by offseting the middle line
    while len(curves) < 3:
        shift = 400
        new_curve = np.copy(curves[mid_lane_index])
        if mid_lane_index == 1:
            # Shifting fixed distance prependicular to the line to the right
            new_curve[1] = shift * np.sqrt(new_curve[0] ** 2 + 1) + new_curve[1]
            curves.append(new_curve)
        else:
            # Shifting fixed distance prependicular to the line to the left
            new_curve[1] = -shift * np.sqrt(new_curve[0] ** 2 + 1) + new_curve[1]
            curves.insert(0, new_curve)
            mid_lane_index += 1

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
    edges = cv2.Canny(cv2.resize(img, None, fx=0.1, fy=0.1), 200, 300)
    kernel = np.ones([3,3])
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Finding all contours in the edge image
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    points = []
    for i in contours:
        # Filtering Contours by area
        area = cv2.contourArea(i)
        if area > 250 or area < 30:
            continue

        # Approximating number of vertices for contours
        epsilon = 0.05*cv2.arcLength(i,True)
        approx = cv2.convexHull(i)
        approx = cv2.approxPolyDP(approx,epsilon,True)

        # Filtering Contours by number of vertices
        if len(approx) >= 4:
            M = cv2.moments(approx)
            points.append((int(M['m10']/M['m00']) * 10, int(M['m01']/ M['m00']) * 10))
    
    # for i, rect in enumerate(rects):
    #     cv2.fillPoly(img, [rect * 10], color=(0, 255, 0))
    #     cv2.circle(img, points[i], 10, (0,0,255), -1)
    
    # cv2.imshow("img", cv2.resize(img, None, fx=0.5, fy=0.5))
    # cv2.imshow("edges", cv2.resize(edges, None, fx=5, fy=5))

    return points

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

def draw_lane(img, curve1, curve2, color=(255, 255, 0)):
    x = np.array([i for i in range(0, img.shape[0], 10)], dtype=np.int)
    y1 = np.polyval(curve1, x).astype(np.int)
    y2 = np.polyval(curve2, x).astype(np.int)
    points = np.row_stack((np.column_stack((y1, x)), np.column_stack((y2, x))[::-1]))
    cv2.fillPoly(img, [points], color=color)

def prespective_transform(img, inverse=False):
    pts1 = np.float32([[0, img.shape[0] // 2], [img.shape[1], img.shape[0] // 2],
                       [110, img.shape[0] // 4], [img.shape[1] - 110, img.shape[0] // 4]])
    pts2 = np.float32([[0, img.shape[0] // 2], [img.shape[1], img.shape[0] // 2],
                       [0, 0], [img.shape[1], 0]])
    
    # result = img
    # for pt in pts1:
    #     cv2.circle(result, (pt[0], pt[1]), 5, (0, 255, 255), thickness=-1)

    if inverse:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply prespective transform and fill background with grey pixels
    grey_img = np.zeros_like(img)
    grey_img[:,:,:] = 100
    result = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), grey_img, borderMode=cv2.BORDER_TRANSPARENT)
    return result

def print_benchmark():    
    print(
        '\n' * 20 + \
        f"Benchmark over {count} frames:\n" + \
        f"Average FPS  = {count / (time.time() - start_time): .2f}\n" + \
        f"Total time   = {time.time() - start_time: .2f}s\n" + \
        f"Frame        = {frame_time: .2f}s\n" + \
        f"Prespective  = {prespective_time: .2f}s\n" + \
        f"lane_point   = {lane_point_time: .2f}s\n" + \
        f"find_cluster = {find_cluster_time: .2f}s\n" + \
        f"get_curves   = {get_curves_time: .2f}s\n" + \
        f"draw_lane    = {draw_lane_time: .2f}s\n" + \
        f"display      = {display_time: .2f}s"
    )

cap = cv2.VideoCapture("sim_vid.mp4")

frame_time = 0
prespective_time = 0
lane_point_time = 0
find_cluster_time = 0
get_curves_time = 0
draw_lane_time = 0
display_time = 0
count = 0
start_time = time.time()

paused = False
while True:
    frame_time -= time.time()
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if frame_number == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()

    img = frame[:frame.shape[0] - 30,:,:]
    frame_time += time.time()

    
    cv2.imshow("imgpresp", cv2.resize(img, None, fx=0.5, fy=0.5))
    prespective_time -= time.time()
    img_presp = prespective_transform(img)
    img = img_presp
    prespective_time += time.time()
    
    lane_point_time -= time.time()
    points = get_lane_points(img_presp)
    lane_point_time += time.time()

    find_cluster_time -= time.time()
    labels = find_clusters(points)
    find_cluster_time += time.time()

    get_curves_time -= time.time()
    curves = get_lane_curves(points, labels, img_presp)
    get_curves_time += time.time()

    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    cluster_img = np.zeros_like(img)
    labels = get_ordered_labels(points, labels)
    for i, p in enumerate(points):
        if labels[i] == -1:
            continue
        class_index = labels[i]
        cv2.circle(cluster_img, (int(p[1]), int(p[0])), 8, colors[class_index % 4], -1)
    
    # for i, curve in enumerate(curves):
    #     img = draw_curve(img, curve, colors[i], thickness=5)

    for i in range(1, len(curves)):
        draw_lane_time -= time.time()
        draw_lane(img, curves[i - 1], curves[i], colors[i - 1])
        draw_lane_time += time.time()

    count += 1
    if count == 1000:
        print_benchmark()

    display_time -= time.time()
    cv2.imshow("clusters", cv2.resize(cluster_img, None, fx=0.5, fy=0.5))

    cv2.imshow("img", cv2.resize(img, None, fx=0.5, fy=0.5))
    
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
    
    display_time += time.time()

cv2.destroyAllWindows()
cap.release()
