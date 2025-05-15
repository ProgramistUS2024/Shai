import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ВАРІАНТ 4
# Генерація випадкових даних
np.random.seed(42)
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))
y = y.ravel()


# Побудова графіка початкових даних
plt.scatter(X.ravel(), y, color='blue', label='Випадкові дані')

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Поліноміальна регресія (ступінь 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Показ значень першої точки
print("X[0]:", X[0])
print("X_poly[0]:", X_poly[0])

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Вивід коефіцієнтів
print("\nПоліноміальна модель:")
print("Коефіцієнти:", poly_reg.coef_)
print("Вільний член:", poly_reg.intercept_)

# Графіки
# Сортування для плавної лінії регресії
X_sorted = np.sort(X, axis=0)
X_poly_sorted = poly_features.transform(X_sorted)
y_poly_sorted = poly_reg.predict(X_poly_sorted)

plt.plot(X, y_lin_pred, color='green', label='Лінійна регресія', linewidth=2)
plt.plot(X_sorted, y_poly_sorted, color='red', label='Поліноміальна регресія (ст.2)', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна та поліноміальна регресія (Варіант 4)')
plt.legend()
plt.grid(True)
plt.show()

# Метрики якості
print("\nМетрики якості поліноміальної регресії:")
print("MAE =", round(mean_absolute_error(y, y_poly_pred), 2))
print("MSE =", round(mean_squared_error(y, y_poly_pred), 2))
print("R2  =", round(r2_score(y, y_poly_pred), 2))
