import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

df= pd.read_csv('anemia.csv')

print("First 5 rows: ")
print(df.head())

print("\n Dataset Info: ")
print(df.info())

print("\n Summary Stats: ")
print(df.describe())

print("\n Missing Values: ")
print(df.isnull().sum())

# checking for the count of anemia and not anemia
results = df['Result'].value_counts()
results.plot(kind = 'bar', color = ['blue', 'green'])
plt.xlabel('Result')
plt.ylabel('Frequency')
plt.title('Count of Result')
plt.show()


majorclass = df[df['Result'] == 0]
minorclass = df[df['Result'] == 1]

major_downsample = resample(majorclass, replace = False, n_sample = len(minorclass), random_state=42)
df = pd.concat([major_downsample, minorclass])
print(df['Result'].value_counts())

sns.set(style="whitegrid")

# Count of Anemia vs Normal
plt.figure(figsize=(6, 4))
sns.countplot(x="Result", data=df, palette='Set2')
plt.title('Class Distribution: Anemia (1) vs Normal (0)')
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()

# Distrbution of Heamoglobin
plt.figure(figsize=(6, 4))
sns.histplot(df['Hemoglobin'], kde=True, color='tomato')
plt.title('Heamoglobin Distribution')
plt.xlabel('Heamoglobin')
plt.show()

# Boxplot of Haemoglobin
plt.figure(figsize=(6, 4))
sns.boxplot(x='Result', y = 'Hemoglobin', data=df)
plt.title('Heamoglobin vs Anemia Result')
plt.show()

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

#bivariate analysis
plt.figure(figsize=(6, 6))
ax = sns.barplot(y=df['Heamoglobin'], x=df['Gender'], hue=df['Result'], ci=None)
ax.set(label=['male', 'female'])
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title("Mean Heamoglobin by Gender and Result")
plt.show()

# Feature Selection
X = df.drop('Result', axis =1) # input feature
Y = df['Result'] # target variable

# Split: 80% training, 20%testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

# Train Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)

acc_lr = accuracy_score(Y_test, y_pred)
c_lr = classification_report(Y_test, y_pred)

print("Accuracy Score: ", acc_lr)
print(c_lr)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
acc_rf = accuracy_score(Y_test, y_pred)
c_rf = classification_report(Y_test, y_pred)
print("Accuracy Score: ", acc_rf)
print(c_rf)

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, Y_train)
y_pred = dt.predict(X_test)
acc_dt = accuracy_score(Y_test, y_pred)
c_dt = classification_report(Y_test, y_pred)
print("Accuracy Score: ", acc_dt)
print(c_dt)

# Gaussian Naive Bayes
gnb  = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred = gnb.predict(X_test)
acc_gnb = accuracy_score(Y_test, y_pred)
c_gnb = classification_report(Y_test, y_pred)
print("Accuracy Score: ", acc_gnb)
print(c_gnb)

# Support Vector Machine
svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
acc_svc = accuracy_score(Y_test, y_pred)
c_svc = classification_report(Y_test, y_pred)
print("Accuracy Score: ", acc_svc)
print(c_svc)

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, Y_train)
y_pred = gbc.predict(X_test)
acc_gbc = accuracy_score(Y_test, y_pred)
c_gbc = classification_report(Y_test, y_pred)
print("Accuracy Score: ", acc_gbc)
print(c_gbc)

# Best model: Random Forest
joblib.dump(rf, "model.pkl")
print("\n Best model (Random Forest) saved as model.pkl")