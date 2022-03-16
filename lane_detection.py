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
    kernel = np.ones([4,1])
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

def get_cluster_class(data, labels):
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
    class_to_index = np.zeros((cluster_count,), dtype=np.int)
    indices = np.argwhere(np.ones_like(centres))
    for a, (c, i) in enumerate(sorted(zip(centres, indices))):
        class_to_index[i] = a

    return class_to_index

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

while True:
    start = time.time()
    _, frame = cap.read()
    
    img = frame[:frame.shape[0] - 30,:,:]
    
    points = get_lane_points(img)

    pre_time = time.time()
    labels = find_clusters(points)
    class_idx = get_cluster_class(points, labels)

    fit_time = time.time()

    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    cluster_img = np.zeros_like(img)
    for i, p in enumerate(points):
        if labels[i] == -1:
            continue
        class_index = class_idx[labels[i]]
        cv2.circle(cluster_img, (int(p[1]), int(p[0])), 8, colors[class_index % 4], -1)
    cv2.imshow("clusters", cluster_img)

    cv2.imshow("img", img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
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
