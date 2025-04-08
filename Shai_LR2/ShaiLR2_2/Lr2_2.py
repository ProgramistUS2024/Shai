import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Завантаження штучного нелінійного набору даних
X, y = datasets.make_moons(n_samples=300, noise=0.25, random_state=42)

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабування ознак
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Типи ядер
kernels = {
    'Поліноміальне ядро (degree=3)': SVC(kernel='poly', degree=3),
    'Гаусове (RBF) ядро': SVC(kernel='rbf'),
    'Сигмоїдальне ядро': SVC(kernel='sigmoid')
}

# Побудова моделей і оцінка якості
for name, model in kernels.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Звіт з метриками
    print(classification_report(y_test, y_pred, digits=4))

    # Матриця плутанини
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(name)
    plt.grid(False)
    plt.show()
