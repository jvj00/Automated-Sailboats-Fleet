import numpy as np

PROB_OF_CONNECTION = 1
N_BOATS = 4
N_MEASUREMENTS = 200
N_MSGS = 200


class Boat:
    def __init__(self) -> None:
        self.result = np.zeros((5, 5))
        self.count_others = np.zeros((5, 5))
        self.sum_others = np.zeros((5, 5))
        self.meas = None
        self.row_meas = None
        self.col_meas = None


boats = []
errors = []
for i in range(N_BOATS):
    boats.append(Boat())
    errors.append([])
map = np.array([
       [100, 200, 300, 400, 500],
       [500, 150, 250, 350, 500],
       [600, 700, 850, 750, 450],
       [550, 100, 200, 300, 100],
       [900, 450, 250, 750, 450]
       ])


for mes in range(N_MSGS):
    # MEASUREMENT IN A RANDOM POSITION
    if mes < N_MEASUREMENTS:
        for i in range(N_BOATS):
            boats[i].row_meas = np.random.randint(0, 5)
            boats[i].col_meas = np.random.randint(0, 5)
            boats[i].meas = map[boats[i].row_meas][boats[i].col_meas] + np.random.normal(0, 5)
    else:
        for i in range(N_BOATS):
            boats[i].meas = None
            boats[i].row_meas = None
            boats[i].col_meas = None
    
    # EXCHANGE MESSAGES
    for i in range(N_BOATS): #receiver
        for row in range(5):
            for col in range(5):
                boats[i].count_others[row][col] = 0
                boats[i].sum_others[row][col] = 0

        for j in range(N_BOATS): #sender
            if i != j and np.random.rand() < PROB_OF_CONNECTION:
                for row in range(5):
                    for col in range(5):
                        if boats[j].result[row][col] != 0:
                            boats[i].count_others[row][col] += 1
                            boats[i].sum_others[row][col] += boats[j].result[row][col]

    # COMPUTE RESULT
    for i in range(N_BOATS):
        for row in range(5):
            for col in range(5):
                # CASE ALL: data from old value, data from measurement, (possible) data from others
                if boats[i].result[row][col] !=0 and boats[i].row_meas == row and boats[i].col_meas == col:
                    boats[i].result[row][col] = 0.5 * (1 - boats[i].count_others[row][col]/N_BOATS) * (boats[i].result[row][col] + boats[i].meas)
                    boats[i].result[row][col] += boats[i].sum_others[row][col] / N_BOATS
                # CASE OLD VALUE AND OTHERS: data from old value, (possible) data from others
                elif boats[i].result[row][col] !=0:
                    boats[i].result[row][col] = (1 - boats[i].count_others[row][col]/N_BOATS) * boats[i].result[row][col]
                    boats[i].result[row][col] += boats[i].sum_others[row][col] / N_BOATS
                # CASE MEASUREMENT AND OTHERS: data from measurement, (possible) data from others
                elif boats[i].row_meas == row and boats[i].col_meas == col:
                    boats[i].result[row][col] = (1 - boats[i].count_others[row][col]/N_BOATS) * boats[i].meas
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