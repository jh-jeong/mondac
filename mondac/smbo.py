import time
import logging

import numpy as np
from ConfigSpace.util import get_one_exchange_neighbourhood

from .acquisitions import acquisition_map
from .domain import ConfigurationDAG
from .objects import MondrianForest


# TODO configuration_dag --> space
class MF_SMBO(MondrianForest):
    def __init__(self, configuration_dag, acquisition='ei', lower_bound=None,
                 size_ensemble=20, split_threshold=1, min_for_search=10, seed=1, debug=False):

        if acquisition not in acquisition_map:
            raise ValueError("Invalid acquisition setting")
        if not isinstance(configuration_dag, ConfigurationDAG):
            raise ValueError(configuration_dag)

        self.configuration_dag = configuration_dag
        self.acquisition = acquisition_map[acquisition]
        self.lower_bound = lower_bound
        self.n_data = 0
        self.configurations = []
        self.acquisition_hyperparameters = {"ei_xi": -0.005,
                                            "ucb_kappa": 2.0}
        self.min_for_search = min_for_search

        self.seed = seed
        random_state = np.random.RandomState(seed)

        self.debug = debug
        self.logger = logging.getLogger("mondac")
        log_level = logging.DEBUG if self.debug else logging.INFO
        self.logger.setLevel(log_level)
        self.logger.addHandler(logging.NullHandler())

        super(MF_SMBO, self).__init__(configuration_dag, random_state, size_ensemble, split_threshold)

    def update_configurations(self, evaluations):
        if len(evaluations) == 0:
            raise ValueError("There should be at least one point to construct MondrianForest")
        for config in evaluations:
            self.configuration_dag.check_configuration(config)

        configurations = []
        y = []
        for conf, score in evaluations.items():
            configurations.append(conf)
            y.append(score)

        if self.lower_bound is not None:
            if any([(l < self.lower_bound) for l in y]):
                raise ValueError("Invalid label is given under the bounded condition of MFO")
            y = [np.log2(l-self.lower_bound) for l in y]

        self.partial_fit(configurations, y)
        self.n_data += len(configurations)
        self.configurations.extend(configurations)

    def select_configuration(self):
        t_start0 = time.time()
        t_start = time.time()

        if self.n_data < self.min_for_search:
            return self.configuration_dag.sample_configuration()

        acquisitions = self.map_acquisition()
        candidates = sorted(enumerate(acquisitions),
                            key=lambda x: x[1], reverse=True)[:5]
        t_end = time.time()
        self.logger.info("Mapping previous points : %s" % (t_end-t_start))
        t_start = time.time()

        optimized = []

        for idx, acq in candidates:
            config = self.configurations[idx]
            config_opt = self._local_search(config, acq)
            optimized.append(config_opt)

        t_end = time.time()
        self.logger.info("Local searching : %s" % (t_end - t_start))
        t_start = time.time()

        randoms = self.configuration_dag.sample_configuration(20)
        acquisitions_random = self.map_acquisition(randoms)
        for config, acq in zip(randoms, acquisitions_random):
            config_opt = self._local_search(config, acq)
            optimized.append(config_opt)

        optimized = sorted(optimized, reverse=True)[0][1]

        t_end = time.time()
        self.logger.info("Random samples ei : %s" % (t_end - t_start))

        self.logger.info("New configuration is selected: %s sec" % (t_end - t_start0))
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