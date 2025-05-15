import matplotlib.pyplot as pltмоделі
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Поділ на тренувальні та тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення та навчання лінійної регресійної моделі
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Прогнозування результатів
y_pred = regr.predict(X_test)

# Вивід коефіцієнтів
print("Коефіцієнти моделі:")
print(regr.coef_)

# Вивід зсуву (intercept)
print("Зсув (intercept):")
print(regr.intercept_)

# Метрики якості моделі
print("\nОцінка моделі:")
print("R2 score:", round(r2_score(y_test, y_pred), 2))
print("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred), 2))

# Побудова графіка "факт" vs "прогноз"
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
ax.set_title('Лінійна регресія: реальні vs передбачені значення')
plt.grid(True)
plt.show()
