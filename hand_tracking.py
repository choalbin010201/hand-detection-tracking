"""Week 7 skeleton completion: IoU + detection-order greedy tracker.

Usage: python hand_tracking.py --source test.mp4
       python hand_tracking.py --self-test
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def compute_iou(box1, box2):
    """xyxy 연속 좌표의 IoU. 면적 0인 박스는 IoU 0으로 처리."""
    a = [float(v) for v in box1]
    b = [float(v) for v in box2]
    for box in (a, b):
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError('Box must contain four finite coordinates.')
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError('Reversed xyxy coordinates.')
    # 교집합이 없을 때 너비/높이가 음수가 되지 않게 한다.
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box.copy() if hasattr(box, 'copy') else list(box)


class Tracker:
    def __init__(self, threshold=0.5):
        if not 0 < threshold <= 1:
            raise ValueError('threshold must be in (0, 1].')
        self.tracks = []
        self.next_id = 0
        self.threshold = threshold

    def update(self, detections):
        updated_tracks = []
        matched_ids = set()
        for box in detections:
            best_track = None
            best_iou = -1.0
            # PDF 스켈레톤과 같이 detection 순서대로 이전 Track 중 최적 후보 탐색.
            for track in self.tracks:
                if track.id in matched_ids:
                    continue
                iou = compute_iou(box, track.box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is not None and best_iou >= self.threshold:
                matched_ids.add(best_track.id)
                updated_tracks.append(Track(best_track.id, box))
            else:
                updated_tracks.append(Track(self.next_id, box))
                self.next_id += 1
        # 이번 프레임에 매칭되지 않은 Track은 제거: 제공된 기본 알고리즘 유지.
        self.tracks = updated_tracks
        return updated_tracks


def visualize(frame, tracks, trajectories):
    import cv2
    active_ids = {track.id for track in tracks}
    for track_id in list(trajectories):
        if track_id not in active_ids:
            del trajectories[track_id]
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.box)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        points = trajectories.setdefault(track.id, [])
        points.append(center)
        trajectories[track.id] = points = points[-30:]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(frame, f'hand ID: {track.id}', (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for p1, p2 in zip(points, points[1:]):
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
    return frame


def self_test():
    import unittest

    class Tests(unittest.TestCase):
        def test_identical(self):
            self.assertEqual(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)

        def test_partial_and_symmetric(self):
            a, b = [0, 0, 10, 10], [5, 0, 15, 10]
            self.assertAlmostEqual(compute_iou(a, b), 1 / 3)
            self.assertEqual(compute_iou(a, b), compute_iou(b, a))

        def test_no_overlap_touch_and_zero(self):
            for b in ([20, 0, 30, 10], [10, 0, 20, 10], [0, 0, 0, 0]):
                self.assertEqual(compute_iou([0, 0, 10, 10], b), 0)
            self.assertEqual(compute_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)

        def test_contained_and_decimal(self):
            self.assertEqual(compute_iou([0, 0, 10, 10], [0, 0, 5, 5]), 0.25)
            self.assertAlmostEqual(compute_iou([0.5, 0.5, 2.5, 2.5], [1.5, 0.5, 3.5, 2.5]), 1 / 3)

        def test_bad_input(self):
            for b in ([3, 0, 1, 1], [0, 0, float('nan'), 1]):
                with self.assertRaises(ValueError):
                    compute_iou([0, 0, 1, 1], b)

        def test_keep_ids_with_reordered_detections(self):
            t = Tracker()
            t.update([[0, 0, 10, 10], [30, 0, 40, 10]])
            self.assertEqual([x.id for x in t.update([[31, 0, 41, 10], [1, 0, 11, 10]])], [1, 0])

        def test_one_to_one_and_new(self):
            t = Tracker()
            t.update([[0, 0, 10, 10]])
            self.assertEqual([x.id for x in t.update([[0, 0, 10, 10], [0, 0, 10, 10]])], [0, 1])

        def test_threshold_inclusive(self):
            t = Tracker(0.5)
            t.update([[0, 0, 10, 10]])
            self.assertEqual(t.update([[0, 0, 5, 10]])[0].id, 0)

        def test_disappearance_and_reappearance(self):
            t = Tracker()
            t.update([[0, 0, 10, 10]])
            self.assertEqual(t.update([]), [])
            self.assertEqual(t.update([[0, 0, 10, 10]])[0].id, 1)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--source', default='test.mp4', help='Recorded video file, not webcam index')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Tracking IoU threshold')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    import os
    os.chdir(Path(__file__).resolve().parent)
    if args.weights is None:
        record = Path('latest_training.json')
        if record.is_file():
            args.weights = json.loads(record.read_text())['best_weights']
        else:
            args.weights = 'runs/detect/hand_train/weights/best.pt'
    import cv2
    import torch
    from ultralytics import YOLO
    for value in (args.source, args.weights):
        if not Path(value).is_file():
            raise FileNotFoundError(f'File not found: {value}')
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = YOLO(args.weights)
    if len(model.names) != 1 or str(model.names[0]).lower() != 'hand':
        raise ValueError('Use the fine-tuned single-class hand model, not COCO weights.')
    cap = cv2.VideoCapture(str(Path(args.source).resolve()))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video. Export as an H.264 MP4 and retry.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError('Video FPS is unavailable. Re-export the video as an MP4.')
    outdir = Path('outputs') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    outdir.mkdir(parents=True)
    output = outdir / 'hand_tracking.mp4'
    writer = None
    tracker = Tracker(args.threshold)
    trajectories = {}
    frame_count = tracked_frames = streak = longest = 0
    previous_ids = set()
    id_streaks = {}
    longest_same_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # 짝수 해상도로 축소하여 MP4 writer 호환성과 처리 효율을 확보.
            h, w = frame.shape[:2]
            scale = min(1.0, 960 / max(h, w))
            size = (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
            frame = cv2.resize(frame, size)
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                if not writer.isOpened():
                    raise RuntimeError('Cannot create MP4 output.')
            result = model(frame, device=device, conf=args.conf, classes=[0], verbose=False)[0]
            tracks = tracker.update(result.boxes.xyxy.cpu().numpy())
            frame_count += 1
            ids = {t.id for t in tracks}
            tracked_frames += bool(ids)
            streak = streak + 1 if ids else 0
            longest = max(longest, streak)
            id_streaks = {i: id_streaks.get(i, 0) + 1 if i in previous_ids else 1 for i in ids}
            longest_same_id = max([longest_same_id, *id_streaks.values()])
            previous_ids = ids
            frame = visualize(frame, tracks, trajectories)
            cv2.putText(frame, f'{frame_count / fps:.1f}s  hands: {len(tracks)}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
            if frame_count % 30 == 0:
                print(f'Processed {frame_count} frames ({frame_count / fps:.1f}s of source)', flush=True)
            if args.show:
                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    if not frame_count:
        raise RuntimeError('No decodable frames in source video.')
    summary = dict(source=str(Path(args.source).resolve()), weights=str(Path(args.weights).resolve()),
                   device=device, fps=fps, frames=frame_count, duration_seconds=frame_count / fps,
                   frames_with_detections=tracked_frames, longest_detected_seconds=longest / fps,
                   longest_same_id_seconds=longest_same_id / fps, allocated_ids=tracker.next_id,
                   tracking_iou_threshold=args.threshold, confidence_threshold=args.conf,
                   output=str(output.resolve()), note='Frame/FPS durations; visually inspect identity and false detections.')
    (outdir / 'tracking_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if longest_same_id / fps < 5:
        print('CHECK: No single ID persisted for 5 seconds. Inspect video and improve detection/recording.')
    print('Saved:', output.resolve())


if __name__ == '__main__':
    main()
