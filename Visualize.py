import Core
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

housing = (Core.strat_split())

housing = pd.DataFrame(list(housing))
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude')
plt.show()