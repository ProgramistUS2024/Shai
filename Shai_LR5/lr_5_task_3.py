from collections import Counter
from fractions import Fraction

# Вхідні дані
data = [
    ['Sunny',    'High',   'Weak',   'No'],
    ['Sunny',    'High',   'Strong', 'No'],
    ['Overcast', 'High',   'Weak',   'Yes'],
    ['Rain',     'High',   'Weak',   'Yes'],
    ['Rain',     'Normal', 'Weak',   'Yes'],
    ['Rain',     'Normal', 'Strong', 'No'],
    ['Overcast', 'Normal', 'Strong', 'Yes'],
    ['Sunny',    'High',   'Weak',   'No'],
    ['Sunny',    'Normal', 'Weak',   'Yes'],
    ['Rain',     'High',   'Weak',   'Yes'],
    ['Sunny',    'Normal', 'Strong', 'Yes'],
    ['Overcast', 'High',   'Strong', 'Yes'],
    ['Overcast', 'Normal', 'Weak',   'Yes'],
    ['Rain',     'High',   'Strong', 'No']
]

# Функція для обчислення ймовірностей
def naive_bayes_predict(query, data):
    total = len(data)
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    
    results = {}
    for label in label_counts:
        prob = Fraction(label_counts[label], total)
        for i in range(len(query)):
            attr_val = query[i]
            matching = [row for row in data if row[i] == attr_val and row[-1] == label]
            count = len(matching)
            label_total = label_counts[label]
            prob *= Fraction(count, label_total)
        results[label] = prob

    total_prob = sum(results.values())
    normalized = {label: float(p / total_prob) for label, p in results.items()}
    return normalized

# 4-й варіант
variant = ['Sunny', 'Normal', 'Strong']
probs = naive_bayes_predict(variant, data)
decision = max(probs, key=probs.get)

print(f"Варіант 4: {variant} =>")
print(f"  Ймовірність Yes: {probs.get('Yes'):.4f}")
print(f"  Ймовірність No:  {probs.get('No'):.4f}")
print(f"  Рішення: Матч {'відбудеться' if decision == 'Yes' else 'не відбудеться'}")
