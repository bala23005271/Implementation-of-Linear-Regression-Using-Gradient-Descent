# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: bala murugan s
RegisterNumber: 212223230027
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('/content/50_Startups.csv',header=None)
print(data.head())


X=(data.iloc[1:, :-2].values)
print(X)


X1=X.astype(float)
scaler=StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)


X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)


theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicition value: {pre}") 

```

## Output:

![Screenshot 2024-08-29 183628](https://github.com/user-attachments/assets/a6757808-d2b5-43e7-84df-7d905921efa9)


![Screenshot 2024-08-29 183902](https://github.com/user-attachments/assets/222eec0a-231f-4868-924c-e5ef5cdb73af)



![Screenshot 2024-08-29 183939](https://github.com/user-attachments/assets/f5e92ce7-0079-41d2-ada2-9d750fd96555)



![Screenshot 2024-08-29 183956](https://github.com/user-attachments/assets/01bbe8f1-6ff8-4a71-9971-467a68c65fa1)


![Screenshot 2024-08-29 184017](https://github.com/user-attachments/assets/1cc455c0-401f-4b95-8eec-3eb2ee9fc80e)


![Screenshot 2024-08-29 183019](https://github.com/user-attachments/assets/fecd0007-9351-4a0c-8df1-dc9ce935a2cd)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
