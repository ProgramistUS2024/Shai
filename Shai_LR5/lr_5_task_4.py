import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

file_path = "data.txt"
df = pd.read_csv(file_path)
df = df.dropna()
df['cheap'] = df['price'].apply(lambda x: 1 if x < 47 else 0)

features = ['train_type', 'origin', 'destination', 'train_class', 'fare']
df = df[features + ['cheap']]

le = LabelEncoder()
for col in features:
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df['cheap']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Результати класифікації ===")
print("Точність:", round(accuracy_score(y_test, y_pred), 4))
print("\nЗвіт класифікації:\n", classification_report(y_test, y_pred))

sample = [[1, 2, 3, 1, 0]]
pred = model.predict(sample)
print("\nПрогноз для нового запису:", "Дешевий квиток (<47€)" if pred[0] == 1 else "Дорогий квиток (>=47€)")
