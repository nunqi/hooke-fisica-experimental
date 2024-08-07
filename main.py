import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

m1_x = np.array([1.7, 3.4, 5.2, 7, 8.6, 10.7, 12, 13.9])
m1_y = np.array([1.01, 2.02, 3.06, 4.04, 5.00, 6.01, 7.02, 8.06])

m2_x = np.array([1, 2, 3.2, 4, 5, 6.1, 7, 8])
m2_y = np.array([1.01, 2.02, 3.06, 4.04, 5.00, 6.01, 7.02, 8.06])


if m1_x.size != m1_y.size or m1_x.size != m2_x.size or m2_x.size != m2_y.size:
    print("Wrong number of elements in the arrays")
    exit(1)


# linear regression
def linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = x.size
    sum_xy = 0
    sum_x2 = 0
    for i in range(n):
        sum_xy += x[i] * y[i]
        sum_x2 += x[i] ** 2
    m = ((n * sum_xy) - (sum(x) * sum(y))) / ((n * sum_x2) - (sum(x) ** 2))
    b = (sum(y) - (m * sum(x))) / n
    return { "m": m, "b": b }

lr1 = linear_regression(m1_x, m1_y)
lr2 = linear_regression(m2_x, m2_y) 

m1_lr_x = m1_x
m1_lr_y = np.zeros(m1_y.size)
m2_lr_x = m2_x
m2_lr_y = np.zeros(m2_y.size)
for i in range(m1_x.size):
    m1_lr_y[i] = lr1["m"] * m1_lr_x[i] + lr1["b"]
    m2_lr_y[i] = lr2["m"] * m2_lr_x[i] + lr2["b"]

# show
print(f'Mola 1: m = {lr1["m"]}, b = {lr1["b"]}, R² = {r2_score(m1_y, m1_lr_y)}')
print(f'Mola 2: m = {lr2["m"]}, b = {lr2["b"]}, R² = {r2_score(m2_y, m2_lr_y)}')

plt.plot(m1_lr_x, m1_lr_y, label=f'Mola 1 - Regressão Linear (R² = {r2_score(m1_y, m1_lr_y)})')
plt.plot(m2_lr_x, m2_lr_y, label=f'Mola 2 - Regressão Linear (R² = {r2_score(m2_y, m2_lr_y)})')
plt.plot(m1_x, m1_y, '*', label='Mola 1 - Dados medidos')
plt.plot(m2_x, m2_y, '*', label='Mola 2 - Dados medidos')
plt.xlabel('Deslocamento (cm)')
plt.ylabel('Força (N)')
plt.legend()
plt.show()
