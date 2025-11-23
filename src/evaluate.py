# evaluate.py
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from model_xception import build_xception
from dataset_tf import make_dataset
from sklearn.preprocessing import LabelEncoder

def gather_image_paths(base, split='test'):
    paths, labels = [], []
    base = Path(base, split)
    for cls in sorted([d.name for d in base.iterdir() if d.is_dir()]):
        for f in (base/cls).glob('*.*'):
            paths.append(str(f))
            labels.append(cls)
    return paths, labels

def main(data_dir, weights_path=None):
    test_paths, test_labels = gather_image_paths(data_dir, 'test')
    le = LabelEncoder()
    y_true = le.fit_transform(test_labels)
    ds = make_dataset(test_paths, y_true, batch_size=32, shuffle=False, img_size=(299,299), augment_on=False)
    model = build_xception(input_shape=(299,299,3), num_classes=1 if len(le.classes_)==2 else len(le.classes_))
    if weights_path:
        model.load_weights(weights_path)

    preds = model.predict(ds)
    if preds.shape[1]==1:
        y_pred = (preds.ravel() > 0.5).astype(int)
    else:
        y_pred = preds.argmax(axis=1)

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <data_dir> [weights_path]")
        sys.exit(1)

    data_dir = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else None

    main(data_dir, weights)

