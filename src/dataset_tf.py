# dataset_tf.py
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def decode_and_resize(filename, label, img_size=(224,224)):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # 0..1
    image = tf.image.resize(image, img_size)
    return image, label

def augment(image, label):
    try:
        import tensorflow_addons as tfa
        angle = tf.random.uniform([], -0.1, 0.1)
        image = tfa.image.rotate(image, angle)
    except Exception:
        pass

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label


def make_dataset(file_paths, labels, batch_size=32, shuffle=True, img_size=(299,299), augment_on=False):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))
    ds = ds.map(lambda f,l: decode_and_resize(f,l,img_size), num_parallel_calls=AUTOTUNE)
    if augment_on:
        try:
            import tensorflow_addons as tfa
        except Exception:
            tfa = None
        ds = ds.map(lambda im,l: (tf.image.random_flip_left_right(im), l), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
