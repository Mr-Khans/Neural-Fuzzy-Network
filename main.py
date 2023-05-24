import numpy as np
import skfuzzy as fuzz
from keras.models import Sequential
from keras.layers import Dense

# Generate random training data
np.random.seed(0)
y_pred_fuzzy = []

X_train = np.random.rand(1000, 3)
y_train = np.random.rand(1000, 1)
# Create fuzzy logic for input variables
x1 = np.arange(0, 1.01, 0.01)
x2 = np.arange(0, 1.01, 0.01)
x3 = np.arange(0, 1.01, 0.01)
# Define membership functions for each variable
x1_lo = fuzz.trimf(x1, [0, 0, 0.5])
x1_md = fuzz.trimf(x1, [0, 0.5, 1])
x1_hi = fuzz.trimf(x1, [0.5, 1, 1])
x2_lo = fuzz.trimf(x2, [0, 0, 0.5])
x2_md = fuzz.trimf(x2, [0, 0.5, 1])
x2_hi = fuzz.trimf(x2, [0.5, 1, 1])
x3_lo = fuzz.trimf(x3, [0, 0, 0.5])
x3_md = fuzz.trimf(x3, [0, 0.5, 1])
x3_hi = fuzz.trimf(x3, [0.5, 1, 1])
# Define fuzzy logic rules for predicting temperature
r1 = np.fmin(np.fmin(x1_lo, x2_lo), x3_lo)
r2 = np.fmin(np.fmin(x1_lo, x2_lo), x3_md)
r3 = np.fmin(np.fmin(x1_lo, x2_md), x3_hi)
r4 = np.fmin(np.fmin(x1_hi, x2_hi), x3_hi)
# Compute membership functions for each rule
out_lo = np.fmax(r1, r2)
out_md = r3
out_hi = r4

print("Low temp : ",out_lo)
print("Mid temp : ",out_md)
print("High temp : ",out_hi)

model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(100, input_dim=3, activation='relu'))
model.add(Dense(50, input_dim=3, activation='relu'))
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=500, batch_size=32)

X_test = np.random.rand(10, 3)
y_pred = model.predict(X_test)
print(X_test)
print(y_pred)

for y in y_pred:
    if y < 0.3:
        y_pred_fuzzy.append(out_lo)
    elif y < 0.7:
        y_pred_fuzzy.append(out_md)
    else:
        y_pred_fuzzy.append(out_hi)
y_pred_fuzzy_mean = np.mean(y_pred_fuzzy)

print("Mean temp: ", y_pred_fuzzy_mean)
print("Low temp : ", out_lo)
print("Mid temp : ", out_md)
print("High temp : ", out_hi)


