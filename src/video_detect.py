# video_detect.py
import cv2
import numpy as np
import tensorflow as tf
from model_xception import build_xception
from pathlib import Path
import matplotlib.pyplot as plt


def frame_generator(video_path, sample_rate=1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % int(fps // sample_rate if fps >= sample_rate else 1) == 0:
            yield frame_no, frame, frame_no / fps
        frame_no += 1
    cap.release()


def detect_periods(preds, times, min_duration=0.5, merge_gap=0.5):
    periods = []
    start = None
    for p, t in zip(preds, times):
        if p and start is None:
            start = t
        elif (not p) and (start is not None):
            end = prev_t
            if end - start >= min_duration:
                periods.append((start, end))
            start = None
        prev_t = t
    if start is not None and t - start >= min_duration:
        periods.append((start, t))

    # merge close gaps
    merged = []
    for s, e in periods:
        if not merged:
            merged.append([s, e])
        else:
            if s - merged[-1][1] <= merge_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
    return merged


def main(video_path, weights_path, threshold=0.5, sample_rate=1):
    model = build_xception(input_shape=(299, 299, 3), num_classes=1)
    model.load_weights(weights_path)

    preds_bool = []
    times = []

    for fn, frame, seconds in frame_generator(video_path, sample_rate=sample_rate):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(img, (299, 299))
        img = tf.cast(img, tf.float32) / 255.0
        img = np.expand_dims(img.numpy(), 0)

        prob = model.predict(img, verbose=0).ravel()[0]

        preds_bool.append(prob < threshold)

        times.append(seconds)

    periods = detect_periods(preds_bool, times)
    print("Detected periods (s):", periods)

    plt.figure(figsize=(12, 2))
    plt.plot(times, preds_bool, drawstyle='steps-post')
    plt.xlabel('Time (s)')
    plt.ylabel('Logo detected (1=Yes, 0=No)')
    plt.title('Logo detection over time')
    plt.ylim(-0.1, 1.1)
    plt.show()

    return periods


if __name__ == '__main__':
    import sys
    video = sys.argv[1]
    weights = sys.argv[2]
    main(video, weights)
