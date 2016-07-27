import copy

import networkx as nx
from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace


class ConfigurationDAG(ConfigurationSpace):
    def __init__(self, seed=1, weight=100):
        super(ConfigurationDAG, self).__init__(seed)

        self.dag = nx.DiGraph()
        self.initial_weight = weight
        self.dag.add_node('__HPOlib_configuration_space_root__', weight=weight)

    def add_hyperparameter(self, hyperparameter):
        super(ConfigurationDAG, self).add_hyperparameter(hyperparameter)
        name = hyperparameter.name
        self.dag.add_node(name, hyperparameter=hyperparameter)
        self.dag.add_edge('__HPOlib_configuration_space_root__', name)
        self._distribute_weights('__HPOlib_configuration_space_root__', self.initial_weight)

    def add_condition(self, condition):
        super(ConfigurationDAG, self).add_condition(condition)

        parent = condition.parent.name
        child = condition.child.name

        if isinstance(condition, EqualsCondition):
            self.dag.add_edge(parent, child)
        elif isinstance(condition, InCondition):
            self.dag.add_edge(parent, child)
        else:
            raise NotImplementedError()

        self.dag.remove_edge('__HPOlib_configuration_space_root__', child)

        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("The condition (%s) made a cycle" % condition)

        # TODO reduce unnecessary edges

        self._distribute_weights('__HPOlib_configuration_space_root__', self.initial_weight)

    def _distribute_weights(self, node, weight):
        # TODO larger weight is prefered?
        self.dag.node[node]['weight'] = weight
        n_child = len(self.dag.successors(node))
        if n_child == 0:
            return
        for n in self.dag.successors(node):
            self._distribute_weights(n, float(weight)/n_child)

    def add_configuration_space(self, prefix, configuration_space,
                                delimiter=":"):
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("The method add_configuration_space must be "
                            "called with an instance of "
                            "HPOlibConfigSpace.configuration_space."
                            "ConfigurationSpace.")

        for hp in configuration_space.get_hyperparameters():
            new_parameter = copy.deepcopy(hp)
            new_parameter.name = "%s%s%s" % (prefix, delimiter,
                                             new_parameter.name)
            self.add_hyperparameter(new_parameter)

        for condition in configuration_space.get_conditions():
            dlcs = condition.get_descendant_literal_conditions()
            for dlc in dlcs:
                if not dlc.child.name.startswith("%s%s" % (prefix, delimiter)):
                    dlc.child.name = "%s%s%s" % (
                        prefix, delimiter, dlc.child.name)
                if not dlc.parent.name.startswith("%s%s" % (prefix, delimiter)):
                    dlc.parent.name = "%s%s%s" % (
                        prefix, delimiter, dlc.parent.name)
            self.add_condition(condition)

        for forbidden_clause in configuration_space.forbidden_clauses:
            dlcs = forbidden_clause.get_descendant_literal_clauses()
            for dlc in dlcs:
                if not dlc.hyperparameter.name.startswith(
                                "%s%s" % (prefix, delimiter)):
                    dlc.hyperparameter.name = "%s%s%s" % \
                                              (prefix, delimiter,
                                               dlc.hyperparameter.name)
            self.add_forbidden_clause(forbidden_clause)

        return configuration_space

