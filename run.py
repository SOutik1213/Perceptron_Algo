import pandas as pd
import numpy as np
from model import Perceptron
from sklearn.metrics import accuracy_score

train=pd.read_csv("./Perceptron_Algo/train.csv")
test=pd.read_csv("./Perceptron_Algo/test.csv")

X_train=np.array(train.iloc[:,:-1])
y_train=np.array(train.iloc[:,-1])
X_test=np.array(test.iloc[:,:-1])
y_test=np.array(test.iloc[:,-1])


pe = Perceptron(lr=0.01, max_iter=100000)
pe.fit(X_train, y_train)

# Predict
preds = pe.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, preds))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Ensure predictions and true labels are numpy arrays of int
y_test = np.array(y_test).astype(int)
preds = np.array(preds).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test, preds)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Perceptron Confusion Matrix on Test Set")
plt.grid(False)
plt.show()





