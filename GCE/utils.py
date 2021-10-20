import os
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shutil import copyfile
import psutil
import gc


def import_from(module, name):
    """
    from 'module' import 'name'
    :param module: module name
    :param name: name of function that shall be imported
    :return: imported function
    """
    if module.endswith(".py"):
        module = module[:-3]
    if "/" in module:
        path_split = os.path.split(module)
        module = path_split[-1]
        sys.path.append(os.path.join(*path_split[:-1]))
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


# Customised dictionary class enabling dot access for convenience
class DotDict(dict):
    def __init__(self, *args, **kwargs):

        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DotDict(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = DotDict(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = DotDict(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def convert_to_dict(self, delete_functions=False):
        out_dict = dict()
        for key in self.keys():
            if isinstance(self[key], DotDict):
                out_dict[key] = self[key].convert_to_dict(delete_functions=delete_functions)
            else:
                if not callable(self[key]) or not delete_functions:
                    out_dict[key] = self[key]
        return out_dict


def backup_one_file(file_in, location):
    """
    This function copies the file file_in to the specified location, with a timestamp added.
    :param file_in: filename
    :param location: backup location
    """
    datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
    file_backup = os.path.join(location, os.path.split(file_in)[-1][:-3] + "_" + datetime + ".py")
    copyfile(file_in, file_backup)


def multipage(filename, figs=None, dpi=360):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        # fig.set_rasterized(True)  # rasterised?
        fig.savefig(pp, format='pdf', dpi=dpi)
    pp.close()


def auto_garbage_collect(pct=80.0):
    """
    This function collects the garbage.
    :param pct: Collect garbage when the memory consumption is higher than pct percent.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
