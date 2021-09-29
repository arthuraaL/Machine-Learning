from utils import *
import matplotlib.pyplot as plt
import numpy as np

fetch_housing_data()
housing = load_housing_data()

# Housing info
#print(df.info())

# Let's look at a summary of the numerical attributes
#print(df.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

def split_train_test(data, test_ratio, random_state=1, shuffle=True):
    np.random.seed(random_state)
    indices = list(range(len(data)))
    
    if shuffle:
        np.random.shuffle(indices)

    test_size = int(len(indices) * test_ratio)
    train_indices, test_indices = indices[test_size:], indices[:test_size]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    x, y = split_train_test(housing, 0.2)
    print(x[0:4])
    print(y[0:4])