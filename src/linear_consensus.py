import numpy as np

PROB_OF_CONNECTION = 1
N_BOATS = 4
N_MSGS = 500
N_MEASUREMENTS = N_MSGS
SIZE_MAP_X = 60
SIZE_MAP_Y = 60


class Boat:
    def __init__(self) -> None:
        self.result = np.zeros((SIZE_MAP_Y, SIZE_MAP_X))
        self.count_others = np.zeros((SIZE_MAP_Y, SIZE_MAP_X))
        self.sum_others = np.zeros((SIZE_MAP_Y, SIZE_MAP_X))
        self.meas = []
    def measure_sonar(self, row, col, value):
        self.meas.append((row, col, value + np.random.normal(0, 5)))
    def get_measure(self, row, col):
        for m in self.meas:
            if m[0] == row and m[1] == col:
                return m[2]
        return 0
    def empty_measures(self):
        self.meas = []


boats = []
errors = []
for i in range(N_BOATS):
    boats.append(Boat())
    errors.append([])
map = np.random.rand(SIZE_MAP_Y, SIZE_MAP_X) * 500 + 100


for mes in range(N_MSGS):
    # MEASUREMENT IN A RANDOM POSITION
    for i in range(N_BOATS):
        boats[i].empty_measures()
    if mes < N_MEASUREMENTS:
        for i in range(N_BOATS):
            n_measure_per_step = np.random.randint(1, 5)
            for _ in range(n_measure_per_step):
                row = np.random.randint(0, SIZE_MAP_Y)
                col = np.random.randint(0, SIZE_MAP_X)
                boats[i].measure_sonar(row, col, map[row][col])
    
    
    # EXCHANGE MESSAGES
    for i in range(N_BOATS): #receiver
        for row in range(SIZE_MAP_Y):
            for col in range(SIZE_MAP_X):
                boats[i].count_others[row][col] = 0
                boats[i].sum_others[row][col] = 0

        for j in range(N_BOATS): #sender
            if i != j and np.random.rand() < PROB_OF_CONNECTION:
                for row in range(SIZE_MAP_Y):
                    for col in range(SIZE_MAP_X):
                        if boats[j].result[row][col] != 0:
                            boats[i].count_others[row][col] += 1
                            boats[i].sum_others[row][col] += boats[j].result[row][col]

    # COMPUTE RESULT
    for i in range(N_BOATS):
        for row in range(SIZE_MAP_Y):
            for col in range(SIZE_MAP_X):
                # CASE ALL: data from old value, data from measurement, (possible) data from others
                if boats[i].result[row][col] !=0 and boats[i].get_measure(row, col) != 0:
                    boats[i].result[row][col] = 0.5 * (1 - boats[i].count_others[row][col]/N_BOATS) * (boats[i].result[row][col] + boats[i].get_measure(row, col))
                    boats[i].result[row][col] += boats[i].sum_others[row][col] / N_BOATS
                # CASE OLD VALUE AND OTHERS: data from old value, (possible) data from others
                elif boats[i].result[row][col] !=0:
                    boats[i].result[row][col] = (1 - boats[i].count_others[row][col]/N_BOATS) * boats[i].result[row][col]
                    boats[i].result[row][col] += boats[i].sum_others[row][col] / N_BOATS
                # CASE MEASUREMENT AND OTHERS: data from measurement, (possible) data from others
                elif boats[i].get_measure(row, col) != 0:
                    boats[i].result[row][col] = (1 - boats[i].count_others[row][col]/N_BOATS) * boats[i].get_measure(row, col)
                    boats[i].result[row][col] += boats[i].sum_others[row][col] / N_BOATS
                # CASE OTHERS: data from others
                elif boats[i].count_others[row][col] != 0:
                    boats[i].result[row][col] = boats[i].sum_others[row][col] / boats[i].count_others[row][col]
    
    # STORE RESULT
    for i in range(N_BOATS):
        errors[i].append(np.mean(np.abs(map-boats[i].result)))
    print(f'Iteration {mes+1} of {N_MSGS}: avg error = {np.mean([errors[b][mes] for b in range(N_BOATS)])}')


import matplotlib.pyplot as plt
plt.figure()
for i in range(N_BOATS):
    plt.plot(errors[i], label=f'boat {i}')
plt.legend()
plt.show()