# sort.py — simple SORT tracker
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
              (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:, :4] = np.eye(4)
        self.kf.R *= 10
        self.kf.P *= 10
        self.kf.Q *= 0.01
        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        self.kf.update(np.array(bbox).reshape((4, 1)))
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape((1, 4))[0]

    def get_state(self):
        return self.kf.x[:4].reshape((1, 4))[0]

class Sort:
    def __init__(self, iou_threshold=0.3):
        self.trackers = []
        self.iou_threshold = iou_threshold

    def update(self, detections):
        trks = []
        to_del = []
        for t in self.trackers:
            pos = t.predict()
            trks.append(pos)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)

        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        # new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        results = []
        for t in self.trackers:
            results.append(np.append(t.get_state(), t.id))

        return np.array(results)

    def associate_detections_to_trackers(self, dets, trks):
        if len(trks) == 0:
            return [], list(range(len(dets))), []

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(matched_indices).T

        unmatched_detections = []
        for d in range(len(dets)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trks)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_detections, unmatched_trackers
