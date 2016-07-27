import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from mondac.utils import sample_split


class Box(object):
    __metaclass__ = ABCMeta

    def __init__(self, weight=0.0):
        self.container = []
        self.count = 0
        self.weight = weight
        self.upper = None
        self.lower = None

    def put(self, element):
        self._check_valid(element)
        self.container.append(element)
        self.count += 1
        if self.upper is None or self.upper < element:
            self.upper = element
        if self.lower is None or self.lower > element:
            self.lower = element

    @abstractmethod
    def size(self):
        raise NotImplementedError()

    @abstractmethod
    def distance(self, x):
        raise NotImplementedError()

    @abstractmethod
    def _check_valid(self, element):
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(self, offset):
        raise NotImplementedError()


class RealBox(Box):
    def __init__(self, weight=0.0):
        super(RealBox, self).__init__(weight)

    def size(self):
        if self.count == 0:
            raise ValueError("Empty box")
        return self.weight * (self.upper - self.lower)

    def distance(self, x):
        self._check_valid(x)
        if self.count == 0:
            raise ValueError("Empty box")
        if self.upper < x:
            return self.weight * (x - self.upper)
        elif self.lower > x:
            return self.weight * (self.lower - x)
        else:
            return 0

    def _check_valid(self, element):
        if not isinstance(element, numbers.Real):
            raise ValueError(element)

    def inverse_transform(self, offset):
        return self.lower + offset / float(self.weight)


class CategoricalBox(Box):
    def __init__(self, n_category, weight=0.0):
        super(CategoricalBox, self).__init__(weight)
        if n_category <= 0:
            raise ValueError("n_category > 0")
        self.n_category = n_category

    def size(self):
        if self.count == 0:
            raise ValueError("Empty box")
        if self.n_category == 1:
            return 0.0
        return self.weight * (self.upper - self.lower) / (self.n_category - 1.0)

    def distance(self, x):
        self._check_valid(x)
        if self.count == 0:
            raise ValueError("Empty box")
        if self.n_category == 1:
            return 0.0
        if self.upper < x:
            return self.weight * (x - self.upper) / (self.n_category - 1.0)
        elif self.lower > x:
            return self.weight * (self.lower - x) / (self.n_category - 1.0)
        else:
            return 0

    def _check_valid(self, element):
        if element not in range(self.n_category):
            raise ValueError(element)

    def inverse_transform(self, offset):
        if self.count == 0:
            raise ValueError("Empty box")
        if self.n_category == 1:
            return self.lower
        return self.lower + offset * (self.n_category - 1.0) / self.weight


class ConfigurationBoxes(object):
    def __init__(self, configuration_space):
        if not isinstance(configuration_space, ConfigurationSpace):
            raise ValueError("Type mismatch: configuration_space, %s != ConfigurationSpace"
                             % type(configuration_space))
        self.boxes = {}
        self.configuration_space = configuration_space

    def put(self, configuration):
        self.configuration_space.check_configuration(configuration)
        idx_active = np.where(~np.isnan(configuration._vector))[0]

        for idx in idx_active:
            configuration_space = configuration.configuration_space
            hp_name = configuration_space.get_hyperparameter_by_idx(idx)
            hp = configuration_space.get_hyperparameter(hp_name)
            box = self.boxes.get(hp_name)
            if box is None:
                if isinstance(hp, CategoricalHyperparameter):
                    box = CategoricalBox(hp._num_choices, 1)
                else:
                    box = RealBox(1)
                self.boxes[hp_name] = box
            box.put(configuration._vector[idx])

    def size(self):
        s = 0
        for hp, box in self.boxes.items():
            s += box.size()
        return s

    def distance(self, configuration):
        idx_active = np.where(~np.isnan(configuration._vector))[0]

        d = 0
        for idx in idx_active:
            configuration_space = configuration.configuration_space
            hp_name = configuration_space.get_hyperparameter_by_idx(idx)
            box = self.boxes.get(hp_name)
            if box is not None:
                d += box.distance(configuration._vector[idx])
        return d

    def draw_cut(self, random_state):
        aligned_keys = self.boxes.keys()
        aligned_sizes = [self.boxes[key].size() for key in aligned_keys]
        cut_idx, offset = sample_split(aligned_sizes, random_state)

        cut_hyperparameter = aligned_keys[cut_idx]
        cut_box = self.boxes[cut_hyperparameter]

        cut_x = cut_box.inverse_transform(offset)
        cut_d = self.configuration_space.get_idx_by_hyperparameter_name(cut_hyperparameter)

        return cut_d, cut_x

    def draw_cut_ext(self, configuration, random_state):
        vector = configuration._vector
        idx_active = np.where(~np.isnan(vector))[0]
        aligned_keys = []
        aligned_sizes = []
        for idx in idx_active:
            configuration_space = configuration.configuration_space
            hp_name = configuration_space.get_hyperparameter_by_idx(idx)
            box = self.boxes.get(hp_name)
            if box is None:
                continue
            aligned_keys.append(hp_name)
            aligned_sizes.append(box.distance(vector[idx]))

        cut_idx, offset = sample_split(aligned_sizes, random_state)
        cut_hyperparameter = aligned_keys[cut_idx]
        cut_box = self.boxes[cut_hyperparameter]

        cut_d = self.configuration_space.get_idx_by_hyperparameter_name(cut_hyperparameter)

        offset_r = cut_box.inverse_transform(offset) - cut_box.lower
        if vector[cut_d] > cut_box.upper:
            cut_x = cut_box.upper + offset_r
        else:
            cut_x = vector[cut_d] + offset_r

        return cut_d, cut_x














