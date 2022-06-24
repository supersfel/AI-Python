import numpy as np
import pandas as pd

i=15500
w_hidden = pd.read_csv(f'W_234710\\{i}epoch_W0.csv').to_numpy()
w_output = pd.read_csv(f'W_234710\\{i}epoch_W1.csv').to_numpy()
df0 = pd.DataFrame(w_hidden.T)
df0.to_csv(f'w_hidden.csv',index=False)
df1 = pd.DataFrame(w_output.T)
df1.to_csv(f'w_output.csv', index=False)

w_hidden = pd.read_csv(f'w_hidden.csv',header=None).to_numpy()
w_output = pd.read_csv(f'w_output.csv',header=None).to_numpy()

print(w_hidden.shape)
print(w_output.shape)