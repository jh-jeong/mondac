import numpy as np


def sample_split(boxes, random_state):
    box_size = np.sum(boxes)
    if box_size == 0:
        raise ValueError("The box is Empty")

    point = random_state.uniform(high=box_size)
    d, offset = 0, 0
    for idx, box in enumerate(boxes):
        if point < box:
            d = idx
            offset = point
            break
        point -= box

    return d, offset


