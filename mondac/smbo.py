import time

import numpy as np
from ConfigSpace.util import get_one_exchange_neighbourhood

from .acquisitions import acquisition_map
from .domain import ConfigurationDAG
from .objects import MondrianForest


# TODO configuration_dag --> space
class MF_SMBO(MondrianForest):
    def __init__(self, configuration_dag, acquisition='ei', lower_bound=None,
                 size_ensemble=20, split_threshold=1, min_for_search=10, seed=1,
                 logger=None):
        if acquisition not in acquisition_map:
            raise ValueError("Invalid acquisition setting")
        if not isinstance(configuration_dag, ConfigurationDAG):
            raise ValueError(configuration_dag)

        self.acquisition = acquisition_map[acquisition]
        self.lower_bound = lower_bound
        self.n_data = 0
        self.configurations = []
        self.seed = seed
        random_state = np.random.RandomState(seed)

        self.acquisition_hyperparameters = {"ei_xi": -0.005,
                                            "ucb_kappa": 2.0}
        self.min_for_search = min_for_search
        self.logger = logger
        self.configuration_dag = configuration_dag

        super(MF_SMBO, self).__init__(configuration_dag, random_state, size_ensemble, split_threshold)

    def update_configurations(self, configurations, labels):
        if type(configurations) is not list:
            raise TypeError("configs must be a list, currently %s" % configurations)
        if type(labels) is not list:
            raise TypeError("labels must be a list, currently %s" % labels)
        if len(configurations) != len(labels):
            raise ValueError("configurations and labels should have same length")
        if len(configurations) == 0:
            raise ValueError("There should be at least one point to construct MondrianForest")
        for config in configurations:
            self.configuration_dag.check_configuration(config)

        if self.lower_bound is not None:
            if any([(l < self.lower_bound) for l in labels]):
                raise ValueError("Invalid label is given under the bounded condition of MFO")
            labels = [np.log2(l-self.lower_bound) for l in labels]

        self.partial_fit(configurations, labels)
        self.n_data += len(configurations)
        self.configurations.extend(configurations)

    def select_configurations(self, size=1):
        t_start0 = time.time()
        t_start = time.time()
        if size < 1:
            return []
        if self.n_data < self.min_for_search:
            samples = self.configuration_dag.sample_configuration(size)
            return samples if size > 1 else [samples]

        acquisitions = self.map_acquisition()
        candidates = sorted(enumerate(acquisitions),
                            key=lambda x: x[1], reverse=True)[:5]
        t_end = time.time()
        self.logger.info("==== Mapping previous ei : %s" % (t_end-t_start))
        t_start = time.time()

        optimized = []

        for idx, acq in candidates:
            config = self.configurations[idx]
            config_opt = self._local_search(config, acq)
            optimized.append(config_opt)

        t_end = time.time()
        self.logger.info("==== Local searching : %s" % (t_end - t_start))
        t_start = time.time()

        randoms = self.configuration_dag.sample_configuration(20)
        acquisitions_random = self.map_acquisition(randoms)
        for config, acq in zip(randoms, acquisitions_random):
            config_opt = self._local_search(config, acq)
            optimized.append(config_opt)

        optimized = [c for v, c in sorted(optimized, reverse=True)[:size]]

        self.map_acquisition(configurations=optimized[:2])

        t_end = time.time()
        self.logger.info("==== Random samples ei : %s" % (t_end - t_start))

        self.logger.info("New %d configurations are selected: %s sec" % (size, t_end - t_start0))
        return optimized

    def map_acquisition(self, configurations=None):
        if configurations is None:
            configurations = self.configurations
        predicts_ = []
        for config in configurations:
            predicts_.append(self.predict_ensemble(config))
        predicts_ = np.array(predicts_)
        means_, vars_ = predicts_.T
        if len(configurations) <= 3:
            self.logger.info("means: %s" % means_)
            self.logger.info("std: %s" % np.sqrt(vars_))

        acquisitions = self.acquisition(means_, vars_, y_min=self.y_min,
                                        **self.acquisition_hyperparameters)
        if len(configurations) <= 3:
            self.logger.info("acq: %s" % acquisitions)
        return acquisitions

    def _local_search(self, configuration, acquisition):
        neighbors = get_one_exchange_neighbourhood(configuration, self.seed)
        acquisitions = self.map_acquisition(neighbors)
        while np.any(acquisitions > acquisition):
            acquisition, configuration = max(zip(acquisitions, neighbors))
            neighbors = get_one_exchange_neighbourhood(configuration, self.seed)
            acquisitions = self.map_acquisition(neighbors)
        return acquisition, configuration