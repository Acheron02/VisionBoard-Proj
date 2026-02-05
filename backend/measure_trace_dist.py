import cv2
import numpy as np

MAX_DISTANCE_PX = 20
MIN_TRACE_AREA = 500
EDGE_SAMPLES = 20
PARALLEL_ANGLE_THRESHOLD = 15  # degrees

# ---------- Helpers ----------
def sample_edge_points(contour, n):
    contour = contour.reshape(-1, 2)
    idxs = np.linspace(0, len(contour)-1, n).astype(int)
    return contour[idxs]

def angle_between_vectors(v1, v2):
    """Angle in degrees between two vectors"""
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0
    cos_theta = np.clip(dot / mag, -1.0, 1.0)
    angle = np.arccos(cos_theta) * 180 / np.pi
    return min(angle, 180 - angle)

# ---------- Main function ----------
def measure_parallel_trace_distances(filtered_contours, pcb_roi_shape, pcb_image=None, offset=(0,0)):
    ox, oy = offset
    H, W = pcb_roi_shape[:2]

    annotated_img = pcb_image.copy() if pcb_image is not None else np.zeros((H,W,3), dtype=np.uint8)
    full_H, full_W = annotated_img.shape[:2]

    trace_mask = np.zeros((full_H, full_W), dtype=np.uint8)
    traces = []

    # ---------- Build trace data ----------
    for idx, cnt in enumerate(filtered_contours):
        if cv2.contourArea(cnt) < MIN_TRACE_AREA:
            continue

        cnt_shifted = cnt + np.array([[ox, oy]])
        cv2.drawContours(trace_mask, [cnt_shifted], -1, 255, cv2.FILLED)

        vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        direction = np.array([vx[0], vy[0]], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        normal1 = np.array([-direction[1], direction[0]], dtype=np.float64)
        normal2 = -normal1
        centroid = np.mean(cnt.reshape(-1, 2), axis=0)

        traces.append({
            "idx": idx,
            "contour": cnt,
            "direction": direction,
            "normals": [normal1, normal2],
            "centroid": centroid,
            "edge_pts": sample_edge_points(cnt, EDGE_SAMPLES)
        })

    # Paint all traces green
    annotated_img[trace_mask>0] = (0,255,0)

    results = []
    coord_logs = []
    seen = set()

    # ---------- Measure clearances ----------
    for i, t1 in enumerate(traces):
        for t2 in traces[i+1:]:
            key = frozenset({t1["idx"], t2["idx"]})
            if key in seen:
                continue

            angle = angle_between_vectors(t1["direction"], t2["direction"])
            if angle > PARALLEL_ANGLE_THRESHOLD:
                continue

            vec_centroids = t2["centroid"] - t1["centroid"]
            vec_centroids /= (np.linalg.norm(vec_centroids) + 1e-6)
            normal = max(t1["normals"], key=lambda n: np.dot(n, vec_centroids))

            min_dist = None
            best_pair = None
            best_meta = None

            for p in t1["edge_pts"]:
                x0, y0 = float(p[0]), float(p[1])
                for d in np.linspace(1, MAX_DISTANCE_PX, MAX_DISTANCE_PX*2):
                    x = x0 + normal[0]*d
                    y = y0 + normal[1]*d
                    if x<0 or x>=W or y<0 or y>=H:
                        break
                    if cv2.pointPolygonTest(t2["contour"], (x,y), False)>=0:
                        intersect_vec = np.array([x-x0, y-y0], dtype=np.float64)
                        intersect_vec /= (np.linalg.norm(intersect_vec)+1e-6)
                        if np.dot(intersect_vec, normal) < 0.7:
                            continue
                        if min_dist is None or d<min_dist:
                            min_dist = d
                            best_pair = ((x0,y0),(x,y))
                            best_meta = {
                                "from_trace": t1["idx"],
                                "to_trace": t2["idx"],
                                "start": (x0+ox, y0+oy),
                                "end": (x+ox, y+oy),
                                "normal": (normal[0], normal[1]),
                                "steps": d
                            }
                        break

            if min_dist is not None:
                seen.add(key)
                results.append((t1["idx"], t2["idx"], float(min_dist)))
                coord_logs.append(best_meta)
                p1, p2 = best_pair
                p1i = (int(round(p1[0]+ox)), int(round(p1[1]+oy)))
                p2i = (int(round(p2[0]+ox)), int(round(p2[1]+oy)))
                cv2.line(annotated_img, p1i, p2i, (0,165,255), 5)
                cv2.circle(annotated_img, p1i, 2, (0,165,255), -1)
                cv2.circle(annotated_img, p2i, 2, (0,165,255), -1)

                print(
                    f"[Trace Clearance] T{best_meta['from_trace']} -> T{best_meta['to_trace']} | "
                    f"start={best_meta['start']} end={best_meta['end']} | "
                    f"steps={best_meta['steps']:.2f} | "
                    f"normal=({best_meta['normal'][0]:.2f},{best_meta['normal'][1]:.2f})"
                )

    return results, annotated_img, coord_logs
