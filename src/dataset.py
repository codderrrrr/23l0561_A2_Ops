import tensorflow as tf

def decode_and_resize(file_name, label):
    img_byte = tf.io.read_file(file_name)
    img = tf.io.decode_image(img_byte, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    img.set_shape([*(224, 224), 3])
    return img, label

def data_augmentation(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img, label

def create_dataset(file_paths, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    if is_train:
        dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_train:
        dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset