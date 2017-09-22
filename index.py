import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

hist_data = np.genfromtxt('aapl.csv', delimiter=',')

# print(type(hist_data))
# print(hist_data.shape)

# Remove first Row and isolate last column
y_hist = hist_data[1:,-1]
# print(y_hist.shape)

# Remove first Row and Remove first and last column
x_hist = hist_data[1:,1:-1]

# Scale X_hist
scaler = MinMaxScaler()
x_hist = scaler.fit_transform(x_hist)
print(x_hist)
# print(x_hist[0])

predict_me = scaler.transform([155.80, 155.80, 152.75, 153.39, 153.02,26958560.])
predict_me = np.array(predict_me)

#random_state_list = [10, 20, 30, 40]

#for random_state in random_state_list:
X_train, X_test, Y_train, Y_test = train_test_split(x_hist, y_hist, test_size=0.7, random_state=42)

# Train the model with Classification type SVM
C_iter = [0.1, 1, 10]

for c_value in C_iter:
    # for degree_value in range(0,10):
        # print ("degree value: " + str(degree_value))
    clf = svm.SVC(kernel='linear', C=c_value)  # , degree=int(degree_value))
    clf.fit(X_train, Y_train)
    print("State: " + str(42) + " - C: " + str(c_value) + " - Score: " + str(clf.score(X_test, Y_test)))
    print("Prediction: " + str(clf.predict(predict_me)))

