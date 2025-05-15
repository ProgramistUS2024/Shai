import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
import pickle

# Завантаження даних з файлу відповідно до варіанту 4
input_file = 'data_regr_4.txt'
data = np.loadtxt(input_file, delimiter=',')

# Розділення даних на ознаку (X) та цільову змінну (y)
X, y = data[:, :-1], data[:, -1]

# Розділення на тренувальні та тестові дані
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Створення та тренування лінійної регресійної моделі
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозування
y_pred = regressor.predict(X_test)

# Побудова графіка
plt.scatter(X_test, y_test, color='blue', label='Фактичні значення')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Прогноз')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія (Варіант 4)')
plt.legend()
plt.grid(True)
plt.show()

# Оцінка моделі
print("Оцінка якості моделі:")
print("MAE:", round(sm.mean_absolute_error(y_test, y_pred), 2))
print("MSE:", round(sm.mean_squared_error(y_test, y_pred), 2))
print("R2:", round(sm.r2_score(y_test, y_pred), 2))

# Збереження моделі
with open('model_variant4.pkl', 'wb') as f:
    pickle.dump(regressor, f)
