import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Варіант 4: генерація даних
np.random.seed(42)
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))
y = y.ravel()

# Функція побудови кривих навчання
def plot_learning_curves(model, X, y, title):
    train_errors, val_errors = [], []
    for m in range(2, len(X) + 1):
        X_train = X[:m]
        y_train = y[:m]
        model.fit(X_train, y_train.ravel())
        y_train_predict = model.predict(X_train)
        y_val_predict = model.predict(X)
        train_errors.append(mean_squared_error(y_train, y_train_predict))
        val_errors.append(mean_squared_error(y, y_val_predict))

    plt.plot(np.sqrt(train_errors), label="Тренувальна помилка")
    plt.plot(np.sqrt(val_errors), label="Валідаційна помилка")
    plt.xlabel("Кількість тренувальних прикладів")
    plt.ylabel("Корінь середньоквадратичної помилки (RMSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Лінійна модель
linear_reg = LinearRegression()
plot_learning_curves(linear_reg, X, y, "Криві навчання — Лінійна регресія")

# Поліноміальна модель (ступінь 2)
polynomial_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(polynomial_reg_2, X, y, "Криві навчання — Поліноміальна регресія (ступінь 2)")

# Поліноміальна модель (ступінь 10)
polynomial_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])
plot_learning_curves(polynomial_reg_10, X, y, "Криві навчання — Поліноміальна регресія (ступінь 10)")
