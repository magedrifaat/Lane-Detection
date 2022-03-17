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
    model = DBSCAN(eps=0.5)
    y = model.fit(x).labels_

    # Treat any cluster with low count as noise
    for label in sorted(set(y)):
        if label == -1:
            continue
        freq = sum([l == label for l in y])
        if freq < SAMPLE_SIZE / 10:
            y[np.argwhere(y == label)] = -1
    
    # Re-number clusters to fill any gaps
    labels = set(y)
    labels.discard(-1)
    for i, label in enumerate(sorted(labels)):
        if i != label:
            y[np.argwhere(y == label)] = i
    
    return y

def get_lane_curves(points, labels, img_shape):
    ordered_labels = get_ordered_labels(points, labels)
    cluster_count = len(set(ordered_labels)) - (1 if -1 in ordered_labels else 0)
    # Distribute the points into separate arrays
    lane_lines = [points[np.where(ordered_labels == i)] for i in range(cluster_count)]
    # Fit a quadratic curve to each array
    curves = []
    for line in lane_lines:
        params = np.polyfit(line[:, 0], line[:, 1], 2)
        params2 = np.polyfit(line[:, 0] / 10, line[:, 1] / 10, 2)
        curve_img = np.zeros((img_shape[0] // 10, img_shape[1] // 10))
        curve_img = draw_curve(curve_img, params2, 255)
        count_before = np.sum(np.nonzero(curve_img))
        points_img = np.zeros((img_shape[0] // 10, img_shape[1] // 10))
        for point in line:
            cv2.circle(points_img, (int(point[1]) // 10, int(point[0]) // 10), 3, 255, -1)
        curve_img = cv2.bitwise_and(cv2.bitwise_not(points_img), curve_img)
        percent = np.sum(np.nonzero(curve_img)) / (count_before+1)
        if percent < 0.4:
            curves.append(params)
    return curves

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

def draw_curve(img, curve_param, color=(0,0,255)):
    x = np.array([i for i in range(0, img.shape[0], 10)])
    y = np.polyval(curve_param, x)
    for i in range(1, len(x)):
        cv2.line(img, (int(y[i]), x[i]), (int(y[i - 1]), x[i - 1]), color=color, thickness=3)
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

cap = cv2.VideoCapture("sim_video.mp4")

paused = False
while True:
    start = time.time()
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if frame_number == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()
    
    img = frame[:frame.shape[0] - 30,:,:]
    
    points = get_lane_points(img)

    pre_time = time.time()
    labels = find_clusters(points)
    curves = get_lane_curves(points, labels, img.shape)
    labels = get_ordered_labels(points, labels)

    fit_time = time.time()

    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    cluster_img = np.zeros_like(img)
    for i, p in enumerate(points):
        if labels[i] == -1:
            continue
        class_index = labels[i]
        cv2.circle(cluster_img, (int(p[1]), int(p[0])), 8, colors[class_index % 4], -1)
        
    # for i, curve in enumerate(curves):
    #     img = draw_curve(img, curve, colors[i])
    for i in range(1, len(curves)):
        draw_lane(img, curves[i - 1], curves[i], colors[i - 1])

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

    
    print(
        "\n" * 20 + \
        f"Preprocess time: {(pre_time - start)*1000: .2f}ms\n" + \
        f"Clustering time: {(fit_time - pre_time)*1000: .2f}ms\n" + \
        f"Display time: {(time.time() - fit_time)*1000: .2f}ms\n" + \
        f"Total time: {(time.time() - start) * 1000: .2f}ms\n" + \
        f"FPS: {int(1/(time.time() - start))}", end=""
    )

cv2.destroyAllWindows()
cap.release()
