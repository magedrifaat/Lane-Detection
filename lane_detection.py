import cv2
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import time

SAMPLE_SIZE = 300
CUTOFF_PERCENT = 0.08

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

def filter_curves(lines, img_shape):
    curves = []
    for i, line in enumerate(lines):
        # fit a straight line to the points
        params = np.polyfit(line[:, 0], line[:, 1], 1)
        # fit a curve to the points rescaled by 1/10
        params2 = np.polyfit(line[:, 0] / 10, line[:, 1] / 10, 2)
        
        # draw the curve scaled by 1/10
        curve_img = np.zeros((img_shape[0] // 10, img_shape[1] // 10))
        curve_img = draw_curve(curve_img, params2, 255)

        # get a count of the number of pixels of the curve
        count_before = np.sum(np.nonzero(curve_img))
        
        # draw the original line by connecting the points on the same scale
        points_img = np.zeros((img_shape[0] // 10, img_shape[1] // 10))
        points_img = draw_curve(points_img, line / 10, 255, thickness=15, point_curve=True)

        # subtract the connected lines from the curve
        curve_img = cv2.bitwise_and(cv2.bitwise_not(points_img), curve_img)

        # calculate the percent of the remaining pixels from original
        percent = np.sum(np.nonzero(curve_img)) / (count_before+1)
        # filter curves with coverage of less than 80%
        if percent < 0.2:
            curves.append(params)
    
    # Handle the case of 4 clusters at the fork
    if len(lines) == 4:
        curves = curves[:2]
    
    # Handle the case of high curve third line at the fork
    if len(curves) == 3:
        # calculate the slope of the middle and third line
        mid_curvature = curves[1][0]
        last_curvature = curves[2][0]
        # Ignore the third line if its slope is too different form the middle
        if abs(abs(mid_curvature) - abs(last_curvature)) / abs(mid_curvature) > 3:
            curves = curves[:2]

    # Recreate the third line by offseting the middle line
    if len(curves) == 2:
        shift = 400
        new_curve = np.copy(curves[-1])
        new_curve[1] = shift + new_curve[1]
        curves.append(new_curve)

    return curves

def get_lane_curves(points, labels, img_shape):
    ordered_labels = get_ordered_labels(points, labels)
    cluster_count = len(set(ordered_labels)) - (1 if -1 in ordered_labels else 0)
    # Distribute the points into separate arrays
    lane_lines = [points[np.where(ordered_labels == i)] for i in range(cluster_count)]
    
    return filter_curves(lane_lines, img_shape)

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

def draw_curve(img, curve, color=(0,0,255), thickness=3, point_curve=False):
    if point_curve:
        # If points are provided directly
        x = curve[:, 0].astype(np.int)
        y = curve[:, 1].astype(np.int)
    else:
        # If parameters are provided calculate the points
        x = np.array([i for i in range(0, img.shape[0], 10)])
        y = np.polyval(curve, x).astype(np.int)

    for i in range(1, len(x)):
        cv2.line(img, (y[i], x[i]), (y[i - 1], x[i - 1]), color=color, thickness=thickness)
    return img

def draw_lane(img, curve1, curve2, color=(255, 255, 0)):
    x = np.array([i for i in range(0, img.shape[0], 10)], dtype=np.int)
    y1 = np.polyval(curve1, x).astype(np.int)
    y2 = np.polyval(curve2, x).astype(np.int)
    points = np.row_stack((np.column_stack((y1, x)), np.column_stack((y2, x))[::-1]))
    cv2.fillPoly(img, [points], color=color)

def get_midlane_points(img):
    edges = cv2.Canny(img, 200, 300)
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    rects = []
    for i in contours:
            epsilon = 0.05*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
            area = cv2.contourArea(i)
            if len(approx) == 4 and area < 10000 and area > 2000:
                rects.append(approx)
    
    cv2.drawContours(img, rects, -1, (0,255,0), 2)

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

cap = cv2.VideoCapture("sim_video.mp4")

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

    prespective_time -= time.time()
    img = prespective_transform(img)
    prespective_time += time.time()
    
    lane_point_time -= time.time()
    points = get_lane_points(img)
    lane_point_time += time.time()

    find_cluster_time -= time.time()
    labels = find_clusters(points)
    find_cluster_time += time.time()

    get_curves_time -= time.time()
    curves = get_lane_curves(points, labels, img.shape)
    get_curves_time += time.time()

    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    # cluster_img = np.zeros_like(img)
    # for i, p in enumerate(points):
    #     if labels[i] == -1:
    #         continue
    #     class_index = labels[i]
    #     cv2.circle(cluster_img, (int(p[1]), int(p[0])), 8, colors[class_index % 4], -1)
    
    # for i, curve in enumerate(curves):
    #     img = draw_curve(img, curve, colors[i])
    for i in range(1, len(curves)):
        draw_lane_time -= time.time()
        draw_lane(img, curves[i - 1], curves[i], colors[i - 1])
        draw_lane_time += time.time()

    count += 1
    if count >= 1000:
        break

    display_time -= time.time()
    # cv2.imshow("clusters", cv2.resize(cluster_img, None, fx=0.5, fy=0.5))

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
    
    display_time += time.time()

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
cv2.destroyAllWindows()
cap.release()
