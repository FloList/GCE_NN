from .generators import PairGeneratorCNNPreGenerated
from .datasets import Dataset


def build_pipeline(params):
    """
    Build the data pipeline.
    :param params: parameter dictionary
    :return: generator and dataset objects for training, validation, and testing data (saved in a dictionary)
    """
    generators = dict()
    generators["train"] = PairGeneratorCNNPreGenerated(params, 0)
    generators["val"] = PairGeneratorCNNPreGenerated(params, 1, settings_dict=generators["train"].settings_dict)
    generators["test"] = PairGeneratorCNNPreGenerated(params, 2, settings_dict=generators["train"].settings_dict)

    dataset = dict()
    dataset["train"] = Dataset(generators["train"], params)
    dataset["val"] = Dataset(generators["val"], params)
    dataset["test"] = Dataset(generators["test"], params)

    return generators, dataset
