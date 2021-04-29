import tensorflow as tf
from sklearn.model_selection import train_test_split

def flip_model_x(model):
    return model[::-1, :, :]

def flip_model_y(model):
    return model[:, ::-1, :]

def flip_model_z(model):
    return model[:, :, ::-1]

@tf.function
def data_augmentation(lr_image, hr_image):
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_x(lr_image), flip_model_x(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_y(lr_image), flip_model_y(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_z(lr_image), flip_model_z(hr_image)
    return lr_image, hr_image

def data_pre_processing(lr_data, hr_data, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((lr_data, hr_data))
    dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(len(lr_data))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset
