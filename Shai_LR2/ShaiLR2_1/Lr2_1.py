import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(data[-1])
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(data[-1])
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoders = []
X_encoded = np.empty(X.shape)

for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoders.append(encoder)
        
# Якщо числові стовпці, треба привести до int
X = X_encoded.astype(int)

# Кодування цільових змінних
label_encoder_y = preprocessing.LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Розбиття на навчальну і тестову вибірку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення SVM-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

# Оцінка точності з використанням F1-міри
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Тестова точка
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

# Кодування тестової точки
input_data_encoded = []
encoder_index = 0
for i in range(len(input_data)):
    if input_data[i].isdigit():
        input_data_encoded.append(int(input_data[i]))
    else:
        encoder = label_encoders[encoder_index]
        input_data_encoded.append(int(encoder.transform([input_data[i]])[0]))
        encoder_index += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# Передбачення
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder_y.inverse_transform(predicted_class)[0])
