import tensorflow as tf
import numpy as np
from ..data_utils import set_batch_dim_recursively, get_fermi_counts


class TFData:
    def __init__(self, data: tf.Tensor, label: list, extra_info=None):
        """
        Default class for Tensorflow data
        :param data: map of photon counts
        :param label: list of tf.Tensors containing the labels
        :param extra_info: additional info (that is not needed for training the NN, but might be useful for analysis)
        """
        self.data = data
        self.label = label
        self.extra_info = extra_info

    def vars(self):
        """
        :return: dictionary with data attributes that are not None
        """
        all_vars = vars(self)
        set_keys = [v is not None for v in all_vars.values()]
        return {k: all_vars[k] for k in np.asarray(list(all_vars.keys()))[set_keys]}


class Dataset(object):
    def __init__(self, generator, params):
        """
        Base class for Tensorflow dataset objects
        :param generator: generator object based on which the dataset will be built
        :param params: parameter dictionary
        """
        self._p = params
        self._g = generator
        rescale_full = self._g.settings_dict["rescale"]
        rescale_compressed = rescale_full[self._g.settings_dict["indices_roi"]]
        self._rescale_compressed_expanded = np.expand_dims(rescale_compressed, 0)

        # Store batch size and prefetch buffer size
        if self._g.train_val_test == 0:
            self.bs = self._p.train["batch_size"]
            self._prefetch_buffer = self._p.train["prefetch_batch_buffer"]
        elif self._g.train_val_test == 1:
            self.bs = self._p.train["batch_size_val"]
            self._prefetch_buffer = self._p.train["prefetch_batch_buffer_val"]
        elif self._g.train_val_test == 2:
            self.bs = 1
            self._prefetch_buffer = 0
        else:
            raise NotImplementedError

        self.ds = self.build_dataset()
        self.ds_with_info = self.build_dataset(extra_info=True)
        self.next_element_np = self.ds.as_numpy_iterator()
        self.next_element_with_info_np = self.ds_with_info.as_numpy_iterator()

    #@tf.function  currently crashes on GPU when using tf.function decorator (TF 2.6)
    def build_dataset(self, extra_info=False):
        """
        Build the dataset for tensorflow using the Dataset class
        :return: The iterator returns a TFData object (dictionary containing "data" and "label")
        """
        label_signature = []
        for ls in self._p.nn["label_shape"]:
            label_signature.append(tf.TensorSpec(shape=ls, dtype=tf.float32))
        output_signature = {"data": tf.TensorSpec(shape=self._p.nn["input_shape"], dtype=tf.float32),
                            "label": tuple(label_signature)}

        if extra_info:
            output_signature["extra_info"] = tf.TensorSpec(shape=(), dtype=tf.float32)

        verbose = 1
        if "db" in self._p.keys():
            if self._p.db["chatterbox_generators"]:
                verbose = 3

        # Get dataset
        dataset = tf.data.Dataset.from_generator(lambda: self._g.get_next_pair(extra_info=extra_info, verbose=verbose),
                                                     output_signature=output_signature)
        dataset = dataset.batch(self.bs)
        if self._prefetch_buffer > 0:
            dataset = dataset.prefetch(self._prefetch_buffer)

        # Correct for exposure correction?
        if self._p.nn["remove_exp"]:
            def remove_exp(ds):
                ds["data"] /= self._rescale_compressed_expanded
                return ds

            dataset = dataset.map(remove_exp)

        return dataset

    def get_samples(self, n_samples):
        """
        Sample generator method
        :param n_samples: number of samples to generate
        :return: data, labels, as produced by the tensorflow pipeline
        """
        out_dict = {}
        for _ in range(int(np.ceil(n_samples / self.bs))):
            next_sample = next(self.next_element_np)
            for k in next_sample.keys():
                if isinstance(next_sample[k], np.ndarray):
                    if k in out_dict.keys():
                        out_dict[k] = np.concatenate([out_dict[k], next_sample[k]], axis=0)  # concat along batch dim.
                    else:
                        out_dict[k] = next_sample[k]
                elif isinstance(next_sample[k], (list, tuple)):
                    if k not in out_dict.keys():
                        out_dict[k] = list(next_sample[k])  # temporarily make list (no item assigment for tuples!)
                    else:
                        for elem in range(len(next_sample[k])):
                            out_dict[k][elem] = np.concatenate([out_dict[k][elem], next_sample[k][elem]], axis=0)
        out_dict = set_batch_dim_recursively(out_dict, n_samples, lists_to_tuples=True)

        return out_dict

    def get_fermi_counts(self):
        """
        Returns the counts in the Fermi map after the same pre-processing as for the training data.
        :return: Processed Fermi counts
        """
        required_keys = ["gen", "data", "nn"]
        assert np.all([k in self._p.keys() for k in required_keys]), \
            "Missing keys! Required keys: {:}, found keys: {:}".format(required_keys, self._p.keys())
        rescale = self._g.settings_dict["rescale"]
        indexes_top = self._g.settings_dict["unmasked_pix"]
        return get_fermi_counts(self._p, indexes_top=indexes_top, rescale=rescale)
