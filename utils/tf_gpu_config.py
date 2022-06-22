import tensorflow as tf


def tf_gpu_config(parameter):
    config = tf.ConfigProto()
    config.log_device_placement = parameter['Gpu_config']['log_device_placement']
    # config.gpu_options.visible_device_list = parameter['Gpu_config']['cuda_visible_devices']
    config.gpu_options.allow_growth = parameter['Gpu_config']['allow_growth']
    config.gpu_options.per_process_gpu_memory_fraction = parameter['Gpu_config']['per_process_gpu_memory_fraction']

    return config