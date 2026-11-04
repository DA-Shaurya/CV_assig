"""CCA 3 computer vision tasks. Originals are read only; outputs are derived copies."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage.feature import hog, peak_local_max
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path, max_side=1000):
    if Path(path).suffix.lower() in ('.heic', '.heif'):
        from pillow_heif import register_heif_opener
        register_heif_opener()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((max_side, max_side))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def save(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def table(path, rows):
    if rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def montage(path, items, columns=3):
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.5), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (title, image) in zip(axes.flat, items):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image,
                  cmap='gray' if image.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def restoration(image, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    gaussian = np.clip(image.astype(float) + rng.normal(0, 25, image.shape), 0, 255).astype('uint8')
    salt_pepper = image.copy()
    noise = rng.random(image.shape[:2])
    salt_pepper[noise < .025] = 0
    salt_pepper[noise > .975] = 255
    rows, panels = [], [('Clean reference', image)]
    for label, noisy in [('gaussian_noise', gaussian), ('salt_pepper_noise', salt_pepper)]:
        versions = [('noisy', noisy), ('median', cv2.medianBlur(noisy, 5)),
                    ('gaussian', cv2.GaussianBlur(noisy, (5, 5), 1.2))]
        for method, result in versions:
            save(out / f'{label}_{method}.png', result)
            panels.append((f'{label}: {method}', result))
            rows.append({'noise': label, 'method': method,
                         'PSNR_dB': round(float(peak_signal_noise_ratio(image, result, data_range=255)), 4),
                         'SSIM': round(float(structural_similarity(image, result, channel_axis=2, data_range=255)), 5)})
    table(out / 'metrics.csv', rows)
    montage(out / 'comparison.png', panels)
    return rows


def segmentation(image, out, foreground):
    out.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if foreground == 'dark' else cv2.THRESH_BINARY
    threshold, binary = cv2.threshold(smooth, 0, 255, mode | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    items = [('Binary (Otsu)', binary), ('Erosion', cv2.erode(binary, kernel)),
             ('Dilation', cv2.dilate(binary, kernel)), ('Opening', opened),
             ('Closing', cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)),
             ('Canny edges', cv2.Canny(gray, 100, 200))]
    distance = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    coords = peak_local_max(distance, min_distance=20, threshold_abs=2,
                            labels=(opened > 0).astype(np.uint8), exclude_border=False)
    markers = np.zeros(gray.shape, dtype=np.int32)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    markers[sure_bg == 0] = 1
    for index, (y, x) in enumerate(coords, start=2):
        cv2.circle(markers, (int(x), int(y)), 2, int(index), -1)
    overlay = image.copy()
    if len(coords):
        markers = cv2.watershed(image.copy(), markers)
        overlay[markers == -1] = (0, 0, 255)
    palette = np.random.default_rng(42).integers(40, 255, (max(2, len(coords)+2), 3), dtype=np.uint8)
    labels = palette[np.clip(markers, 0, len(palette)-1)]
    labels[markers <= 1] = 0
    items += [('Distance transform', cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')),
              ('Watershed labels', labels), ('Watershed boundaries', overlay)]
    for i, (title, result) in enumerate(items):
        save(out / f'{i+1:02d}_{title.split(" (")[0].lower().replace(" ", "_")}.png', result)
    montage(out / 'comparison.png', items)
    return {'otsu_threshold': threshold, 'foreground': foreground, 'watershed_seeds': len(coords),
            'note': 'Regions are algorithmic segments, not verified object counts. Use touching objects on a contrasting background and inspect over/under-segmentation.'}


def match_pair(first, second, name, out):
    sift = cv2.SIFT_create(nfeatures=2000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)
    good = []
    if d1 is not None and d2 is not None and len(d2) >= 2:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
    inliers = None
    if len(good) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.sum())
    result = cv2.drawMatches(first, k1, second, k2, good[:60], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save(out / f'{name}_matches.png', result)
    return {'experiment': name, 'keypoints_reference': len(k1), 'keypoints_target': len(k2),
            'ratio_test_matches': len(good), 'RANSAC_inliers': inliers,
            'inlier_ratio': round(inliers / len(good), 4) if inliers is not None and good else None}


def features(image, second, out):
    out.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vector, visual = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    visual = cv2.normalize(visual, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    save(out / 'hog.png', visual)
    np.save(out / 'hog_features.npy', vector)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=2000).detectAndCompute(gray, None)
    save(out / 'sift_keypoints.png', cv2.drawKeypoints(image, keypoints, None,
         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    if descriptors is not None:
        np.save(out / 'sift_descriptors.npy', descriptors)
    montage(out / 'features.png', [('Input', image), ('HOG', visual),
             ('SIFT keypoints', cv2.imread(str(out / 'sift_keypoints.png')))])
    rows = []
    # A transformation is an additional robustness experiment, never a second original photo.
    target = second if second is not None else image
    if second is not None:
        rows.append(match_pair(image, second, 'second_original', out))
    rotated = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
    resized = cv2.resize(target, None, fx=.7, fy=.7, interpolation=cv2.INTER_AREA)
    prefix = 'second' if second is not None else 'self_test'
    for name, transformed in [(f'{prefix}_rotated_90', rotated), (f'{prefix}_resized_70_percent', resized)]:
        save(out / f'{name}.png', transformed)
        rows.append(match_pair(image, transformed, name, out))
    table(out / 'matching_metrics.csv', rows)
    return {'hog_length': len(vector), 'sift_keypoints': len(keypoints), 'matches': rows,
            'second_original_available': second is not None}


def detection(images, out, weights, skip_yolo=False):
    out.mkdir(parents=True, exist_ok=True)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if cascade.empty():
        raise RuntimeError('OpenCV face cascade could not be loaded')
    model = None
    if not skip_yolo:
        from ultralytics import YOLO
        model = YOLO(weights)
    rows = []
    for label, image in images:
        started = time.perf_counter()
        faces = cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(30, 30))
        haar_ms = (time.perf_counter() - started) * 1000
        haar = image.copy()
        for x, y, w, h in faces:
            cv2.rectangle(haar, (x, y), (x+w, y+h), (0, 200, 0), 2)
            cv2.putText(haar, 'face', (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 0), 2)
        save(out / f'{label}_haar.png', haar)
        rows.append({'input': label, 'method': 'Haar face cascade', 'detections': len(faces),
                     'labels': 'face' if len(faces) else 'none', 'milliseconds': round(haar_ms, 2)})
        panels = [('Input', image), ('Haar: faces only', haar)]
        if model is not None:
            started = time.perf_counter()
            result = model.predict(image, conf=.25, device='cpu', verbose=False)[0]
            elapsed = (time.perf_counter() - started) * 1000
            annotated = result.plot()
            save(out / f'{label}_yolo.png', annotated)
            boxes = [{'class': result.names[int(b.cls.item())], 'confidence': round(float(b.conf.item()), 4),
                      'xyxy': b.xyxy[0].tolist()} for b in result.boxes]
            (out / f'{label}_yolo_boxes.json').write_text(json.dumps(boxes, indent=2), encoding='utf-8')
            rows.append({'input': label, 'method': 'YOLO', 'detections': len(boxes),
                         'labels': ', '.join(b['class'] for b in boxes) or 'none', 'milliseconds': round(elapsed, 2)})
            panels.append(('YOLO: supported object classes', annotated))
        montage(out / f'{label}_comparison.png', panels)
    table(out / 'detection_metrics.csv', rows)
    return rows


def motion(video, out):
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video has no valid FPS metadata; convert a copy to a standard MP4 first.')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
    previous = None
    previous_frame = None
    rows, samples = [], []
    writer = None
    i = 0
    try:
        while True:
            ok, original = cap.read()
            if not ok:
                break
            scale = min(1, 640 / original.shape[1])
            w, h = int(original.shape[1]*scale)//2*2, int(original.shape[0]*scale)//2*2
            frame = cv2.resize(original, (w, h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if writer is None:
                writer = cv2.VideoWriter(str(out / 'motion_comparison.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w*2, h*2))
                if not writer.isOpened():
                    raise RuntimeError('Could not initialize MJPG video output')
            t = time.perf_counter()
            raw_foreground = subtractor.apply(frame)
            gmm = np.uint8(raw_foreground == 255) * 255  # Exclude shadow value 127.
            gmm_ms = (time.perf_counter()-t)*1000
            diff = np.zeros_like(gray)
            flow = frame.copy()
            speeds, diff_ms, flow_ms = [], 0., 0.
            if previous is not None:
                t = time.perf_counter()
                diff = cv2.threshold(cv2.absdiff(gray, previous), 25, 255, cv2.THRESH_BINARY)[1]
                diff_ms = (time.perf_counter()-t)*1000
                t = time.perf_counter()
                p0 = cv2.goodFeaturesToTrack(previous, maxCorners=150, qualityLevel=.01, minDistance=10)
                if p0 is not None:
                    p1, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, p0, None, winSize=(21, 21), maxLevel=3)
                    if p1 is not None and status is not None:
                        for start, end in zip(p0[status.ravel()==1], p1[status.ravel()==1]):
                            a, b = start.ravel(), end.ravel()
                            if np.isfinite(b).all():
                                speeds.append(float(np.linalg.norm(b-a)))
                                cv2.arrowedLine(flow, tuple(a.astype(int)), tuple(b.astype(int)), (0, 255, 0), 1, tipLength=.3)
                flow_ms = (time.perf_counter()-t)*1000
            panels = [('Original', frame), ('Frame difference', cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)),
                      ('GMM foreground', cv2.cvtColor(gmm, cv2.COLOR_GRAY2BGR)), ('Lucas-Kanade vectors', flow)]
            labeled = []
            for title, view in panels:
                view = view.copy()
                cv2.rectangle(view, (0, 0), (w, 28), (0, 0, 0), -1)
                cv2.putText(view, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                labeled.append(view)
            writer.write(np.vstack([np.hstack(labeled[:2]), np.hstack(labeled[2:])]))
            if i in {0, max(1, frame_count//4), max(2, frame_count//2), max(3, 3*frame_count//4)}:
                save(out / f'frame_{i:05d}.png', frame)
                save(out / f'frame_{i:05d}_difference.png', diff)
                save(out / f'frame_{i:05d}_gmm.png', gmm)
                save(out / f'frame_{i:05d}_flow.png', flow)
                montage(out / f'frame_{i:05d}_comparison.png', panels, columns=2)
                if previous_frame is not None:
                    save(out / f'frame_{i:05d}_previous.png', previous_frame)
                samples.append((f'frame_{i:05d}', frame))
            if previous is not None:
                rows.append({'frame': i, 'time_seconds': round(i/fps, 3),
                             'difference_foreground_percent': round(100*np.count_nonzero(diff)/diff.size, 3),
                             'GMM_foreground_percent': round(100*np.count_nonzero(gmm)/gmm.size, 3),
                             'tracked_points': len(speeds), 'mean_motion_px_per_frame': round(float(np.mean(speeds)), 4) if speeds else None,
                             'difference_ms': round(diff_ms, 3), 'GMM_ms': round(gmm_ms, 3), 'LK_ms': round(flow_ms, 3)})
            previous, previous_frame = gray, frame.copy()
            i += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if i < 2:
        raise ValueError('At least two readable video frames are required')
    table(out / 'motion_metrics.csv', rows)
    summary = {'frames_processed': i, 'fps': fps, 'duration_seconds': i/fps,
               'duration_within_assignment_range': 15 <= i/fps <= 30,
               'warmup_frames_excluded_from_summary': min(30, len(rows))}
    measured = rows[30:]
    for key in ['difference_foreground_percent', 'GMM_foreground_percent', 'mean_motion_px_per_frame', 'difference_ms', 'GMM_ms', 'LK_ms']:
        values = [r[key] for r in measured if r[key] is not None]
        summary[key] = round(float(np.mean(values)), 4) if values else None
    return summary, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo', default='data/originals/photo1.png')
    parser.add_argument('--second-photo', default='data/originals/photo2.jpg')
    parser.add_argument('--video')
    parser.add_argument('--segmentation-photo', help='Optional original photo with touching objects')
    parser.add_argument('--foreground', choices=['dark', 'light'], default='light')
    parser.add_argument('--weights', default='yolov8n.pt')
    parser.add_argument('--skip-yolo', action='store_true', help='Partial run only; not a complete submission')
    parser.add_argument('--output', default='outputs/run')
    args = parser.parse_args()
    out = Path(args.output)
    if (out / 'results.json').exists():
        parser.error('Output already contains results. Choose a new --output folder to preserve prior evidence.')
    out.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(42)
    image = load_image(args.photo)
    save(out / 'input_photo.png', image)
    second = load_image(args.second_photo) if args.second_photo else None
    if second is not None:
        save(out / 'input_second_photo.png', second)
    manifest = {}
    for name in ['photo', 'second_photo', 'video', 'segmentation_photo']:
        path = getattr(args, name)
        if path:
            with open(path, 'rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest[name] = {'file': Path(path).name, 'bytes': Path(path).stat().st_size, 'sha256': checksum}
    result = {'inputs': manifest, 'processing': 'EXIF orientation applied to derived copy; images limited to 1000 px longest side. Originals unchanged.',
              'restoration': restoration(image, out/'task1_restoration'),
              'segmentation': segmentation(load_image(args.segmentation_photo) if args.segmentation_photo else image, out/'task1_segmentation', args.foreground),
              'features': features(image, second, out/'task2_features'), 'pending': []}
    if args.segmentation_photo:
        save(out / 'input_segmentation_photo.png', load_image(args.segmentation_photo))
    if second is None:
        result['pending'].append('Second original photograph required; transformed self-matches are supplementary only.')
    detection_inputs = [('photo', image)]
    if args.video:
        result['motion'], samples = motion(args.video, out/'task4_motion')
        detection_inputs.extend(samples)
        if not result['motion']['duration_within_assignment_range']:
            result['pending'].append('Video is outside the approximately 15-30 second requirement; replace with a compliant original recording.')
    else:
        result['pending'].append('Original 15-30 second video required for motion and video recognition tasks.')
    try:
        result['detections'] = detection(detection_inputs, out/'task3_detection', args.weights, args.skip_yolo)
    except Exception as error:
        result['pending'].append(f'Detection incomplete: {type(error).__name__}: {error}')
        result['detections'] = []
    if args.skip_yolo:
        result['pending'].append('YOLO skipped; rerun with YOLO before submission.')
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Results saved to {out.resolve()}')
    for pending in result['pending']:
        print('PENDING:', pending)


if __name__ == '__main__':
    main()
