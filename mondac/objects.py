import numpy as np

from .box import ConfigurationBoxes
from .domain import ConfigurationDAG


# FIXME parallelize
# TODO Debug mode
class MondrianNode(object):
    __slots__ = ('parent', 'left', 'middle', 'right', 'cut_d', 'cut_x',
                 't_birth', 'configurations','configuration_boxes', 'y', 'm', 'v')

    def __init__(self, configuration_dag, configurations, y, parent):
        self.configurations = configurations
        self.y = y
        self.configuration_boxes = ConfigurationBoxes(configuration_dag)
        self.t_birth = np.inf
        self.parent = parent
        self.left, self.middle, self.right = None, None, None
        self.cut_d, self.cut_x = None, None

        if len(configurations) == 0:
            raise ValueError("MondrianNode must contain at least one point")
        if len(configurations) != len(y):
            raise ValueError("length mismatch: %d != %d" % (len(configurations), len(y)))

        for config in configurations:
            self.configuration_boxes.put(config)

        self.m = np.mean(self.y)
        self.v = np.mean((self.y - self.m) ** 2)

    def split_node(self):
        if self.cut_d is None:
            raise ValueError("Splitting point should be set before call split_box")

        configurations_l, configurations_m, configurations_r = [], [], []
        y_l, y_m, y_r = [], [], []
        for idx, config in enumerate(self.configurations):
            vec = config._vector
            val = vec[self.cut_d]
            if np.isnan(val):
                configurations_m.append(config)
                y_m.append(self.y[idx])
            elif val <= self.cut_x:
                configurations_l.append(config)
                y_l.append(self.y[idx])
            else:
                configurations_r.append(config)
                y_r.append(self.y[idx])

        if len(configurations_l) * len(configurations_r) == 0:
            raise ValueError("indices are not properly splitted")

        node_l = MondrianNode(self.configuration_boxes.configuration_dag,
                              configurations_l, y_l, self)
        node_r = MondrianNode(self.configuration_boxes.configuration_dag,
                              configurations_r, y_r, self)
        node_m = None
        if len(configurations_m) > 0:
            node_m = MondrianNode(self.configuration_boxes.configuration_dag,
                                  configurations_m, y_m, self)

        self.left = node_l
        self.middle = node_m
        self.right = node_r

        return node_l, node_m, node_r

    def distance(self, configuration):
        return self.configuration_boxes.distance(configuration)


class MondrianTree(object):
    def __init__(self, configuration_dag, configurations, y, split_threshold, random_state):
        self.random_state = random_state
        self.epsilon = MondrianNode(configuration_dag, configurations, y, None)
        self.epsilon.t_birth = 0
        self.root = MondrianNode(configuration_dag, configurations, y, self.epsilon)
        self.split_threshold = split_threshold
        self.grow_tree(self.root)

    def grow_tree(self, node):
        box_total = node.configuration_boxes.size()
        if len(node.configurations) <= self.split_threshold or box_total == 0:
            return

        cut_time = self.random_state.exponential(1./box_total)
        node.t_birth = node.parent.t_birth + cut_time

        cut_d, cut_x = node.configuration_boxes.draw_cut(self.random_state)

        node.cut_d = cut_d
        node.cut_x = cut_x

        node_l, node_m, node_r = node.split_node()
        self.grow_tree(node_l)
        self.grow_tree(node_r)
        if node_m is not None:
            self.grow_tree(node_m)

    def seed_point(self, configuration, label, node):
        dist = node.distance(configuration)
        vector = configuration._vector
        cut_time = self.random_state.exponential(np.divide(1., dist))
        parent = node.parent

        if parent.t_birth + cut_time < node.t_birth:
            cut_d_, cut_x_ = node.configuration_boxes.draw_cut_ext(configuration, self.random_state)

            new_parent = MondrianNode(node.configuration_boxes.configuration_dag,
                                      node.configurations+[configuration],
                                      node.y+[label], parent)
            #new_sibling = MondrianNode(node.configuration_boxes.configuration_dag,
            #                           [configuration], [label], new_parent)
            #node.parent = new_parent

            new_parent.t_birth = parent.t_birth + cut_time
            new_parent.cut_d = cut_d_
            new_parent.cut_x = cut_x_

            #if vector[cut_d_] <= cut_x_:
            #    new_parent.left = new_sibling
            #    new_parent.right = node
            #else:
            #    new_parent.left = node
            #    new_parent.right = new_sibling

            if parent.left == node:
                parent.left = new_parent
            elif parent.right == node:
                parent.right = new_parent
            elif parent.middle == node:
                parent.middle = new_parent
            elif parent == self.epsilon:
                self.root = new_parent
            else:
                raise NotImplementedError("Something weird error occurred during update the parent of MF.\n"
                                          "Current node: %s\n"
                                          "Parent node: %s\n"
                                          "New parent node: %s\n" % (node.t_birth,
                                                                     parent.t_birth,
                                                                     new_parent.t_birth))

            node_l, node_m, node_r = new_parent.split_node()
            self.grow_tree(node_l)
            self.grow_tree(node_r)
            if node_m is not None:
                self.grow_tree(node_m)
            del node

        else:
            node.configurations.append(configuration)
            node.y.append(label)

            node.configuration_boxes.put(configuration)
            node.m = np.mean(node.y)
            node.v = np.mean((node.y - node.m) ** 2)

            if not self.is_leaf(node):
                val = vector[node.cut_d]
                if np.isnan(val):
                    if node.middle is None:
                        new_sibling = MondrianNode(node.configuration_boxes.configuration_dag,
                                                   [configuration], [label], node)
                        node.middle = new_sibling
                        return
                    else:
                        child = node.middle
                elif vector[node.cut_d] <= node.cut_x:
                    child = node.left
                else:
                    child = node.right
                return self.seed_point(configuration, label, child)

    def is_leaf(self, node):
        return np.isinf(node.t_birth)

    def is_root(self, node):
        return node.parent == self.epsilon

    def predict(self, configuration):
        cur = self.root
        prob_not_separated, weight = 1.0, 1.0
        mean_, var_ = 0, 0

        while True:
            delta = cur.t_birth - cur.parent.t_birth
            eta = cur.distance(configuration)
            prob_may_separate = max(0, 1 - np.exp(-delta*eta))

            weight = prob_not_separated * prob_may_separate
            mean_ += weight * cur.m
            var_ += weight * (cur.v + cur.m**2)

            if self.is_leaf(cur):
                weight = prob_not_separated * (1 - prob_may_separate)
                mean_ += weight * cur.m
                var_ += weight * (cur.v + cur.m**2)
                break

            value = configuration._vector[cur.cut_d]
            if np.isnan(value):
                if cur.middle is None:
                    weight = prob_not_separated * (1 - prob_may_separate)
                    mean_ += weight * cur.m
                    var_ += weight * (cur.v + cur.m ** 2)
                    break
                else:
                    cur = cur.middle
            elif value <= cur.cut_x:
                cur = cur.left
            else:
                cur = cur.right

            prob_not_separated *= (1 - prob_may_separate)

        var_ = max(0, var_ - mean_**2)
        return mean_, var_


class MondrianForest(object):
    def __init__(self, configuration_dag, random_state, n_ensemble=20, split_threshold=1):
        self.n_ensemble = n_ensemble
        self.trees = []
        self.y_min = np.inf
        self._is_fitted = False
        self.split_threshold = split_threshold
        self.random_state = random_state
        self.configuration_dag = configuration_dag

        if n_ensemble <= 0:
            raise ValueError("n_ensemble must be positive")
        if not isinstance(configuration_dag, ConfigurationDAG):
            raise ValueError(configuration_dag)

    def _add_point(self, configuration, label):
        if label < self.y_min:
            self.y_min = label
        for tree in self.trees:
            tree.seed_point(configuration, label, tree.root)

    def fit(self, configurations, y):
        if len(configurations) != len(y):
            raise ValueError("data and y must have the same length.")
        for config in configurations:
            self.configuration_dag.check_configuration(config)

        for _ in range(self.n_ensemble):
            tree = MondrianTree(self.configuration_dag,
                                configurations, y,
                                self.split_threshold, self.random_state)
            self.trees.append(tree)
        self._is_fitted = True

    def partial_fit(self, configurations, y):
        if len(configurations) != len(y):
            raise ValueError("data and y must have the same length.")

        if not self._is_fitted:
            return self.fit(configurations, y)

        for idx, config in enumerate(configurations):
            self._add_point(config, y[idx])

    def predict_trees(self, configuration):
        predicts = map(lambda tree: tree.predict(configuration), self.trees)
        return np.array(predicts).T

    def predict_ensemble(self, configuration):
        m_, v_ = self.predict_trees(configuration)
        m2_ = m_**2
        expected_y = np.mean(m_)
        variance_y = np.mean(v_+m2_) - expected_y**2
        return [expected_y, variance_y]
