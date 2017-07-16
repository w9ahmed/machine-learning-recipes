import numpy as np
import matplotlib.pyplot as plt


# Population
greyhounds = 500
labradors = 500

# Heigh
greyhound_height = 28 + 4 * np.random.randn(greyhounds)
labrador_height = 24 + 4 * np.random.randn(labradors)


plt.hist([greyhound_height, labrador_height], stacked=True, color=['r', 'b'])
plt.show()
