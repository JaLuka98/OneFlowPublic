# Script that illustrates that this correlating with mcerp works nicely

import numpy as np
import mcerp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
startTime = datetime.now()

np.random.seed(42)

num_samples = 100

x1 = np.random.exponential(1.0, num_samples)
x2 = np.random.normal(0, 2, num_samples)
x3 = np.random.normal(0, 3, num_samples)
x4 = np.random.normal(0, 4, num_samples)

# Create an index, for example, a range of integers from 0 to the length of the data
index = range(len(x1))

df = pd.DataFrame(
    {'x1': x1,
     'x2': x2,
     'x3': x3,
     'x4': x4
    }, index=index)

# Visualize initial correlations with a pair plot
sns.set(style="ticks")
sns.pairplot(df, kind="scatter")
plt.suptitle("Initial Correlations", y=1.02)
plt.savefig('Corr_before.png')

x123 = df[['x1', 'x2', 'x3']]

print('Corrmatrix before:', np.corrcoef(x123.values.T))

c = np.array([[1.0, 0.9, 0.3],
              [0.9, 1.0, 0.6],
              [0.3, 0.6, 1.0]])

eigenvalues, _ = np.linalg.eig(c)
print(eigenvalues)

correlated_numbers = mcerp.induce_correlations(x123.values, c)

print('Corrmatrix after:', np.corrcoef(correlated_numbers.T))

df[['x1', 'x2', 'x3']] = correlated_numbers

# Visualize initial correlations with a pair plot
sns.set(style="ticks")
sns.pairplot(df, kind="scatter")
plt.suptitle("Initial Correlations", y=1.02)
plt.savefig('Corr_after.png')

print(datetime.now() - startTime)