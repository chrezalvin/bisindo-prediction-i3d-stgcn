import sys

sys.path.extend(['../'])
from graph import tools

num_node = 46
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 2), (2, 3), (3, 4), # body pose
    (5, 6), (6, 7), (7, 8), (8, 9), (5, 10), (10, 11), (11, 12), (12, 13), (10, 14), (14, 15), (15, 16), (16, 17), (18, 19), (19, 20), (20, 21), (18, 22), (22, 23), (23, 24), (24, 25), (5, 22), # left hand
    # connector from body to left hand
    (3, 5),
    (26, 27), (27, 28), (28, 29), (29, 30), (26, 31), (31, 32), (32, 33), (33, 34), (31, 35), (35, 36), (36, 37), (37, 38), (39, 40), (40, 41), (41, 42), (39, 43), (43, 44), (44, 45), (45, 46), (26, 43), # right hand
    # connector from body to right hand
    (4, 26),
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)