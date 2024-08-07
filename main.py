import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

"""
Peça Grande (PG) = 50
Peça 1      (P1) = 9,8
Peça 2      (P2) = 10,4
Peça 3      (P3) = 10,1
Peça 4      (P4) = 10,1

No código, o peso precisa ser em N.
Por isso, os valores finais estão multiplicados
por 10 nos arrays
"""

"""
Mola 1 (grande) - 34 cm inicial
  P4                (10,1)
    valor = 32,3
    deslocamento = 34 - 32,3    = 1,7
  P4 + P3           (20,2)
    valor = 30,6
    deslocamento = 34 - 30,6    = 3,4
  P4 + P3 + P2      (30,6)
    valor = 28,8
    deslocamento = 34 - 28,8    = 5,2
  P4 + P3 + P2 + P1 (40,4)
    valor = 27
    deslocamento = 34 - 27      = 7
  PG                (50)
    valor = 25,4
    deslocamento = 34 - 25,4    = 8,6
  PG + P4           (60,1)
    valor = 23,3
    deslocamento = 34 - 23,3    = 10,7
  PG + P4 + P3      (70,2)
    valor = 22
    deslocamento = 34 - 22      = 12
  PG + P4 + P3 + P2 (80,6)
    valor = 20,1
    deslocamento = 34 - 20,1    = 13,9
"""

m1_x = np.array([1.7, 3.4, 5.2, 7, 8.6, 10.7, 12, 13.9])
m1_y = np.array([1.01, 2.02, 3.06, 4.04, 5.00, 6.01, 7.02, 8.06])


"""
Mola 2 (pequena) - 38,4 cm inicial
  P4                (10,1)
    valor = 37,4
    deslocamento = 38,4 - 37,4  = 1
  P4 + P3           (20,2)
    valor = 36,4
    deslocamento = 38,4 - 36,4  = 2
  P4 + P3 + P2      (30,6)
    valor = 35,2
    deslocamento = 38,4 - 35,2  = 3,2
  P4 + P3 + P2 + P1 (40,4)
    valor = 34,4
    deslocamento = 38,4 - 34,4  = 4
  PG                (50)
    valor = 33,4
    deslocamento = 38,4 - 33,4  = 5
  PG + P4           (60,1)
    valor = 32,3
    deslocamento = 38,4 - 32,3  = 6,1
  PG + P4 + P3      (70,2)
    valor = 31,4
    deslocamento = 38,4 - 31,4  = 7
  PG + P4 + P3 + P2 (80,6)
    valor = 30,4
    deslocamento = 38,4 - 30,4  = 8
"""

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
