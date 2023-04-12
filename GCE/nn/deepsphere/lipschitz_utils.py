import tensorflow as tf

# Group sort activation function
class GroupSort(tf.keras.layers.Layer):
    def __init__(self, n_groups):
        super(GroupSort, self).__init__()
        self.n_groups = n_groups

    def call(self, x):
        return self.group_sort(x, self.n_groups)

    def get_config(self):
        config = super(GroupSort, self).get_config()
        config.update({'n_groups': self.n_groups, 'axis': self.axis})
        return config

    def get_sorting_shape(self, x, n_groups):
        shape = x.shape.as_list()
        num_features = shape[-1]
        if num_features % n_groups:
            raise ValueError(f"number of features({num_features}) needs to be a multiple of n_groups({n_groups})")
        n_per_group = num_features // n_groups
        shape = [-1, n_groups, n_per_group]
        return tf.convert_to_tensor(shape)


    def group_sort(self, x, n_groups):
        if x.shape[0] == 0:
            return x

        size = self.get_sorting_shape(x, n_groups)
        grouped_x = tf.reshape(x, size)
        sorted_x = tf.sort(grouped_x, axis=-1)
        sorted_x = tf.reshape(sorted_x, tf.shape(x))
        return sorted_x



# For testing
# import numpy as np
# x = tf.convert_to_tensor(np.arange(64, 0, -1).reshape(4, 16))
# out = GroupSort(n_groups=4)(x)