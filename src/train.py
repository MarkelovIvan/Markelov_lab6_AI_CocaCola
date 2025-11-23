# train.py
import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_xception import build_xception
from dataset_tf import make_dataset
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def gather_image_paths(data_dir, split='train'):
    paths = []
    labels = []
    base = Path(data_dir, split)
    for cls in sorted([d.name for d in base.iterdir() if d.is_dir()]):
        for f in (base/cls).glob('*.*'):
            paths.append(str(f))
            labels.append(cls)
    return paths, labels

def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le

def main(args):
    # конфіг
    img_size = (299,299)
    input_shape = img_size + (3,)
    batch_size = args.batch_size
    epochs = args.epochs

    train_paths, train_labels = gather_image_paths(args.data_dir, 'train')
    val_paths, val_labels = gather_image_paths(args.data_dir, 'val')

    y_train, le = encode_labels(train_labels)
    y_val = le.transform(val_labels)

    train_ds = make_dataset(train_paths, y_train, batch_size=batch_size, img_size=img_size, augment_on=True)
    val_ds = make_dataset(val_paths, y_val, batch_size=batch_size, img_size=img_size, augment_on=False)

    # модель
    model = build_xception(input_shape=input_shape, num_classes=1 if len(le.classes_)==2 else len(le.classes_), load_imagenet_weights=args.load_imagenet)

    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")

    optimizer = Adam(learning_rate=args.lr)
    if args.mixed_precision:
        pass

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy' if len(le.classes_)==2 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(os.path.join(args.checkpoint_dir, 'best.h5'), monitor='val_accuracy', save_best_only=True, save_weights_only=False)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              callbacks=[ckpt, rl, es])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data', help='path to data with train/val/test subfolders')
    parser.add_argument('--checkpoint_dir', default='../weights', help='where to save model')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--load_imagenet', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    args = parser.parse_args()
    main(args)
