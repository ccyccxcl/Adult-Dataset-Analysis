import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import seaborn as sns
import math
import matplotlib.pyplot as plt
original_data=pd.read_csv("C:\\Users\\ccy\\Desktop\\Adult dataset analysis\\adult.data.txt",names=['Age','Workclass','fnlwgt','Education','Education_Num','Matial Status','Ocaaupation','Relationship','Race','Sex','Capital Gain','Cpital Loss','Hours per week','Country','Target'],sep=r'\s*,\s*',engine='python')
original_data.dropna(how="all",inplace=True)
original_data.tail()
fig=plt.figure(figsize=(20,15))
cols=5
rows=math.ceil(float(original_data.shape[1]/cols))
for i,column in enumerate(original_data.columns):
	ax=fig.add_subplot(rows,cols,i+1)
	ax.set_title(column)
	if original_data.dtypes[column]==np.object:
		original_data[column].value_counts().plot(kind="bar",axes=ax)
	else:
		original_data[column].hist(axes=ax)
		plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.9, wspace=0.2)		
plt.savefig("C:\\Users\\ccy\Desktop\\Adult dataset analysis\\a1.png")
plt.show()
(original_data["Country"].value_counts()/original_data.shape[0]).head()

def number_encode_features(df):
	result=df.copy()
	encoders={}
	for column in result.columns:
		if result.dtypes[column]==np.object:
			encoders[column]=preprocessing.LabelEncoder()
			result[column]=encoders[column].fit_transform(result[column])
	return result,encoders
encoded_data,_=number_encode_features(original_data)
sns.heatmap(encoded_data.corr(),square=True)
plt.savefig("C:\\Users\\ccy\Desktop\\Adult dataset analysis\\a2.png")
plt.show()
original_data[["Education","Education_Num"]].head(15)
original_data[["Sex","Relationship"]].head(15)
encoded_data,encoders=number_encode_features(original_data)
for i,column in enumerate(encoded_data.columns):
	ax=fig.add_subplot(rows,cols,i+1)
	ax.set_title(column)
	encoded_data[column].hist(axes=ax)
	plt.xticks(rotation="vertical")
plt.savefig("C:\\Users\\ccy\Desktop\\Adult dataset analysis\\a3.png")
plt.show()
X_train,X_test,y_train,y_test=cross_validation.train_test_split(encoded_data.drop('Target',1),encoded_data["Target"],train_size=0.7)
scaler=preprocessing.StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train.astype("float64")),columns=X_train.columns)
X_test=scaler.transform(X_test.astype("float64"))
cls=linear_model.LogisticRegression()
cls.fit(X_train,y_train)
y_pred=cls.predict(X_test)
cm=metrics.confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm,annot=True,fmt="d",xticklabels=encoders["Target"].classes_,yticklabels=encoders["Target"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score:%f" % skl.metrics.f1_score(y_test,y_pred))
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.vaules.sort()
plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.savefig("C:\\Users\\ccy\Desktop\\Adult dataset analysis\\a4.png")
plt.show()
binary_data = pd.get_dummies(original_data)
binary_data["Target"] = binary_data["Target_>50K"]
del binary_data["Target_<=50K"]
del binary_data["Target_>50K"]
plt.subplots(figsize=(20,20))
sns.heatmap(binary_data.corr(), square=True)
plt.savefig("C:\\Users\\ccy\Desktop\\Adult dataset analysis\\a5.png")
plt.show()
X_train,X_test,y_train,y_test=cross_validation.train_test_split(binary_data.drop('Target',1),binary_data["Target"],train_size=0.7)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)
cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["Target"].classes_, yticklabels=encoders["Target"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ("F1 score: %f" % skl.metrics.f1_score(y_test, y_pred))
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.values.sort()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.savefig("C:\\Users\\ccy\Desktop\\Adult dataset analysis\\a6.png")
plt.show()
print(cls.coef_)