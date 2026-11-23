import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

values = np.arange(1, 101)

data = {x2: [4*x1 - 2*x2 for x1 in values] for x2 in values}
df = pd.DataFrame(data, index=x1, columns=x2)
df.index.name = 'x1\\x2'
#print(df.head())
x1_coords = df.index.values
x2_coords = df.columns.values
Y_values = df.values

X1_grid, X2_grid = np.meshgrid(x1_coords, x2_coords)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X1_grid, X2_grid, Y_values, cmap='viridis', edgecolor='none')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Y')
ax.set_title(f'График функции 4x1-2x2')

fig.colorbar(surf, shrink=0.5, aspect=10, label='Значение Y')
plt.show()
#---------------------------------------------------------
noise = np.random.uniform(low=0.01, high=0.1)
#print(Y_values[:1])
Y_values = df.values + noise
#print(Y_values[:1])
X1_grid, X2_grid = np.meshgrid(x1_coords, x2_coords)
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111, projection='3d')

surf = ax1.plot_surface(X1_grid, X2_grid, Y_values, cmap='viridis', edgecolor='none')

ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('Y')
ax1.set_title(f'График функции 4x1-2x2(с шумом)')
#df.to_csv('data.csv', sep=';')

dfx = pd.DataFrame({
    'x1': np.repeat(df.index.values, len(df.columns)),
    'x2': np.tile(df.columns.values, len(df.index)),
    'y': Y_values.flatten()
})
dfx.to_csv('dataf.csv', sep=';')

# %matplotlib inline
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt

# A, B = 4, -2
# x1 = np.arange(1, 101)
# x2 = np.arange(1, 101)
# X1, X2 = np.meshgrid(x1, x2, indexing="ij")

# Y = A * X1 + B * X2
# table = pd.DataFrame(Y, index=x1, columns=x2)
# table.index.name = "x1\\x2"
# table.to_csv("lab2_task3_table.csv")


# def plot3d(Z, title):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(X1, X2, Z, cmap="viridis")
#     ax.set_xlabel("x1")
#     ax.set_ylabel("x2")
#     ax.set_zlabel("Y")
#     ax.set_title(title)
#     plt.show()


# plot3d(Y, "Y = 4*x1 - 2*x2")

# # Шум в интервале [0.01; 0.1]
# rng = np.random.default_rng(0)
# noise = rng.uniform(0.01, 0.1, size=Y.shape)
# Y_noisy = Y + noise
# plot3d(Y_noisy, "Y с шумом [0.01; 0.1]")

# # Три столбца: x1, x2, y
# data = pd.DataFrame({"x1": X1.ravel(), "x2": X2.ravel(), "y": Y_noisy.ravel()})
# data.to_csv("lab2_task3_data.csv", index=False)
# print(data.head())
