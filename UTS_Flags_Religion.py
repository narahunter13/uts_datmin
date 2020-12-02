# UTS DMKM 221810473

# Import library
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data',
                      names = ["name", "landmass", "zone", "area", "population",
                                            "language", "religion", "bars", "stripes", "colours",
                                            "red", "green", "blue", "gold", "white", "black", "orange",
                                            "mainhue", "circles", "crosses", "saltires", "quarters", "sunstars",
                                            "crescent", "triangle", "icon", "animate", "text", "topleft",
                                            "botright"])

#Hapus kolom name, topleft, dan botright
dataset = dataset.drop(columns = ["name", "topleft", "botright"])

#Coding untuk data kategorik tapi sangat berpengaruh
le = LabelEncoder()
dataset["mainhue"] = le.fit_transform(dataset["mainhue"])

X = dataset.iloc[:, lambda x : x.columns != "religion"].values
y = dataset.iloc[:, 5].values

# Membagi dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Menggunakan algoritma naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confussion Matrix dan Akurasi
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
