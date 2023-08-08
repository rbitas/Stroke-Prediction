# Stroke-Prediction
**Project 4(Final Project)** <br/>
Machine Learning is the practice of applying algorithms and statistics to create models to create models that can learn from data and then make decisions or predictions about future data. Many of today's leading companies, such as Facebook, Google and Uber, make machine learning a central part of their operations as it automates repetitive tasks, increases efficiency of processes, and mitigates human error which allows employees to focus on more critical or strategic decisions. <br/>

### Many organizations make use of Machine Learning (ML). Some examples include:
1. Image Recognition :camera:  <br/>
* ML can be used to recognizes objects in an image such as faces, abnormalities in health and medicine, image recognition in shopping, and even security and safety. In all of these cases, ML's algorithm lets it's user focus on solving the critical problems at hand such as a farmer intervening when ML notices that a plant has a new disease or a doctor is able to work on finding a solution when an imageing machine notices an abnormality from a previous image. <br/>
2. Predict Traffic Patterns :vertical_traffic_light:   <br/>
* ML can be used to help generate predictions regarding the fastest route to a destination. This is apparent in most GPS programs which take user information from multiple drivers to find the best route. <br/>
3. Fraud Detection :x:  <br/>
* ML can be utilized to analyze user behaviors and patterns which can in turn detect any number of anamolies. For example, if it detects multiple purchases in a short period of time or purchases from a completely different country/IP address, it can warn the user. <br/>
4. Chatbots/Customer Service :speech_balloon: <br/>
* ML can be leveraged to create a chatbot which answers any customer inquiries/problems and provide this support 24/7. This cuts on labour costs as well as creating an easier method for the customer to get help any time of the day. <br/>
5. Data Analysis/Data Entry :computer: <br/>
* ML can help automate the data input side and provide accurate insight and prediction to inform important decision making. Some markets that may use this are health, finance, and marketing. <br/>

## **Project Description**<br/> 
In this project, we are utilizing Machine Learning to predict the likelihood of someone having a stroke based on various health factors including hypertension, heart disease, average glucose level, and bmi and social factors including if married, type of area they live in, and type of work. Although we are not able to accurately present the exact percentage of likelihood of getting a stroke, we are able to say whether one is more likely or less likely to get a stroke given the data.<br/>
![](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3BuaTM0dmhxdjNzbXE2NmE5bm03aDgzdHhuemxvamdtcHJuZzUxbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/iPj5oRtJzQGxwzuCKV/giphy.gif)

## **Dashboard** 
Here is an interactive Dashboard for you to explore our data as well as additional perspectives you can use to see how we analyzed our information. <br/>
   * [Tableau Dashboard](https://public.tableau.com/app/profile/andrei.tabatchouk/viz/Project4ML/MalevFemaleDiseaseCountperType?publish=yes)<br/>

## **Presentation**
Here is the presentation to be accompanied with this project for your viewing pleasure. <br/>
   * [Google Slides Presentation](https://docs.google.com/presentation/d/1Y0AWo-qqOz6cqIyUKf_oMHhe51R879zPMlMUvZuqsuw/edit?usp=sharing)

## **Deployment**
StrokePrediction.py is available for deployment. You may test trial this app through your VSCode terminal with <br/>
   * python StrokePrediction.py.

## **Data** :woman_technologist:
For our project, we have visualized data extracted from the following dataset available in the Resources folder <br/>
   * [healthcare-dataset-stroke-data.csv](https://github.com/rbitas/Stroke-Prediction/blob/main/Data/healthcare-dataset-stroke-data.csv) <br/>

## **How to Run**
**DATA PREPARATION**
<details><summary>Full Directions and coding on how to prepare and run your data</summary>
(full code and notebooks exist in "Machine_Learning_Exploration" folder)

First we encoded our non-numeric variables.

```python
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(df[df_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names_out(df_cat)
encode_df.head()
```

```python
# Merge one-hot encoded features and drop the originals
df = df.merge(encode_df,left_index=True, right_index=True)
df = df.drop(df_cat,1)
df.head()
```

While we started with a very clean dataset overall, there were still things we needed to adjust including dropping the "id" column which is of no use to us and filling the null values in the bmi column.
```python
df = df.drop(columns=['id'])
```

For null bmi values we made the decision to take the mean of the other values and fill the nulls with a number. We decided to use the mean of the all the non-null values.
```python
# Find the mean of the "bmi" column to use as replacement for null values.
df["bmi"].mean()
```
```python
# Replace null values
df['bmi'] = df['bmi'].fillna(29.88)
df.head()
```
```python
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

Now we are ready to split the "stroke" column off as our target or "y" variable and turn our remaining columns into our features or the "X" variables.

```python
# Remove stroke dtarget from features data
y = df.stroke.values
X = df.drop(columns="stroke")

columns = X.columns

X = X.values

# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

ANALYSIS:
After preprocessing our data and testing it in a machine learning model, a 95% accuracy score signaled promising potential for both our data and the initial model. However, a deeper dive into the machine learning reports tells us that is not at all the case. If only 5% of our data set was in the "had stroke" category, the model could predict that no one would have a stroke and achieve 95% accuracy. With this in mind, we realized some kind of oversampling in our model could help with our unbalanced data set.

```python
pip install -U imbalanced-learn
```
```python
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_train, y_train.ravel())
```



Finally, we scaled our data, and then started testing different supervised machine learning and neural network models.

```python
# Preprocess numerical data for neural network

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_resampled)
X_test_scaled = X_scaler.transform(X_test)
```
**Logistic Regression**
```python
X_train_scaled.shape
```
```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='lbfgs',
                                max_iter=200)
classifier
```
```python
classifier.fit(X_train_scaled, y_resampled)
```
```python
predictions = classifier.predict(X_test_scaled)
results = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)
```
```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```
```python
# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)

# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)
```

```python
# Displaying results
print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))
```
```
Confusion Matrix
   Predicted 0	Predicted 1
Actual 0	886	330
Actual 1	13	49
Accuracy Score : 0.7316118935837246
Classification Report
              precision    recall  f1-score   support

           0       0.99      0.73      0.84      1216
           1       0.13      0.79      0.22        62

    accuracy                           0.73      1278
   macro avg       0.56      0.76      0.53      1278
weighted avg       0.94      0.73      0.81      1278
```

**Random Forest**
```python
# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=500)
```
```python
# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_resampled)
```
```
Confusion Matrix
   Predicted 0	Predicted 1
Actual 0	1198	18
Actual 1	58	4
Accuracy Score : 0.9405320813771518
Classification Report
              precision    recall  f1-score   support

           0       0.95      0.99      0.97      1216
           1       0.18      0.06      0.10        62

    accuracy                           0.94      1278
   macro avg       0.57      0.52      0.53      1278
weighted avg       0.92      0.94      0.93      1278
```
```python
# Random Forests in sklearn will automatically calculate feature importance
importances = rf_model.feature_importances_

# We can sort the features by their importance
sorted(zip(rf_model.feature_importances_, columns), reverse=True)
```
```
[(0.3586269837202005, 'age'),
 (0.19090176591868344, 'avg_glucose_level'),
 (0.16707080498112015, 'bmi'),
 (0.03171998311673814, 'ever_married_Yes'),
 (0.029536213658467918, 'ever_married_No'),
 (0.029188100907300233, 'hypertension'),
 (0.019229224656013018, 'heart_disease'),
 (0.018497447842601464, 'smoking_status_never smoked'),
 (0.016788312234218505, 'work_type_Self-employed'),
 (0.016074590019140224, 'smoking_status_formerly smoked'),
 (0.016051747988863208, 'work_type_Private'),
 (0.01413799690174771, 'Residence_type_Rural'),
 (0.014061067324401759, 'Residence_type_Urban'),
 (0.013851618220876286, 'gender_Female'),
 (0.013713634394472358, 'gender_Male'),
 (0.013308747976911962, 'smoking_status_smokes'),
 (0.012785946787004303, 'smoking_status_Unknown'),
 (0.012556788195042062, 'work_type_Govt_job'),
 (0.011823320592265388, 'work_type_children'),
 (7.454391473848977e-05, 'work_type_Never_worked'),
 (1.1606491928524725e-06, 'gender_Other')]
 ```
 ```python
importances_df = pd.DataFrame(sorted(zip(rf_model.feature_importances_, columns), reverse=True))
importances_df.set_index(importances_df[1], inplace=True)
importances_df.drop(columns=1, inplace=True)
importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
importances_sorted = importances_df.sort_values(by='Feature Importances')
importances_sorted.plot(kind='barh', color='lightgreen', title= 'Features Importances', legend=False)
```
<img width="778" alt="Screenshot 2023-08-07 at 6 19 49 PM" src="https://github.com/rbitas/Stroke-Prediction/assets/125224990/880da716-0407-456a-8c32-c8cddb51c660">


**Decision Tree**
```python
# Creating the decision tree classifier instance
model = tree.DecisionTreeClassifier()
```
```python
# Fitting the model
model = model.fit(X_train_scaled, y_resampled)
```
```python
# Making predictions using the testing data
predictions = model.predict(X_test_scaled)
```
```
Confusion Matrix
   Predicted 0	Predicted 1
Actual 0	1172	44
Actual 1	52	10
Accuracy Score : 0.9248826291079812
Classification Report
              precision    recall  f1-score   support

           0       0.96      0.96      0.96      1216
           1       0.19      0.16      0.17        62

    accuracy                           0.92      1278
   macro avg       0.57      0.56      0.57      1278
weighted avg       0.92      0.92      0.92      1278
```
**KerasTuner**
```python
# Create a method that creates a new Sequential model with hyperparameter options
def create_model(hp):
    nn_model = tf.keras.models.Sequential()

    # Allow kerastuner to decide which activation function to use in hidden layers
    activation = hp.Choice('activation',['relu','tanh'])

    # Allow kerastuner to decide number of neurons in first layer
    nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units',
        min_value=1,
        max_value=20,
        step=5), activation=activation, input_dim=21))

    # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
    for i in range(hp.Int('num_layers', 1, 2)):
        nn_model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
            min_value=5,
            max_value=20,
            step=5),
            activation=activation))

    nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Compile the model
    nn_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

    return nn_model
```
```
pip install -q -U keras-tuner
```
```python
# Import the kerastuner library
import keras_tuner as kt

tuner = kt.Hyperband(
    create_model,
    objective="val_accuracy",
    max_epochs=10,
    hyperband_iterations=2)
```
```python
# Run the kerastuner search for best hyperparameters
tuner.search(X_train_scaled,y_resampled,epochs=10,validation_data=(X_test_scaled,y_test))
```
```
Trial 55 Complete [00h 00m 07s]
val_accuracy: 0.7363067269325256

Best val_accuracy So Far: 0.8122065663337708
Total elapsed time: 00h 04m 32s
```
```python
# Get top 3 model hyperparameters and print the values
top_hyper = tuner.get_best_hyperparameters(3)
for param in top_hyper:
    print(param.values)
```
```
{'activation': 'tanh', 'first_units': 16, 'num_layers': 2, 'units_0': 15, 'units_1': 15, 'tuner/epochs': 10, 'tuner/initial_epoch': 4, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '0047'}
{'activation': 'tanh', 'first_units': 16, 'num_layers': 1, 'units_0': 10, 'units_1': 15, 'tuner/epochs': 10, 'tuner/initial_epoch': 4, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '0043'}
{'activation': 'tanh', 'first_units': 11, 'num_layers': 2, 'units_0': 10, 'units_1': 10, 'tuner/epochs': 10, 'tuner/initial_epoch': 4, 'tuner/bracket': 2, 'tuner/round': 2, 'tuner/trial_id': '0037'}
```
```python
# Evaluate the top 3 models against the test dataset
top_model = tuner.get_best_models(3)
for model in top_model:
    model_loss, model_accuracy = model.evaluate(X_test_scaled,y_test,verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
```
40/40 - 0s - loss: 0.4106 - accuracy: 0.8122 - 281ms/epoch - 7ms/step
Loss: 0.41061288118362427, Accuracy: 0.8122065663337708
40/40 - 0s - loss: 0.4391 - accuracy: 0.7746 - 284ms/epoch - 7ms/step
Loss: 0.43909767270088196, Accuracy: 0.7746478915214539
40/40 - 0s - loss: 0.4566 - accuracy: 0.7645 - 292ms/epoch - 7ms/step
Loss: 0.4565829038619995, Accuracy: 0.7644757628440857
```
```python
# Get second best model hyperparameters
second_hyper = tuner.get_best_hyperparameters(2)[1]
second_hyper.values
```
```
{'activation': 'tanh',
 'first_units': 16,
 'num_layers': 1,
 'units_0': 10,
 'units_1': 15,
 'tuner/epochs': 10,
 'tuner/initial_epoch': 4,
 'tuner/bracket': 1,
 'tuner/round': 1,
 'tuner/trial_id': '0043'}
```
```python
# Compare the performance to the second-best model
second_model = tuner.get_best_models(2)[1]
model_loss, model_accuracy = second_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
```
40/40 - 0s - loss: 0.4391 - accuracy: 0.7746 - 324ms/epoch - 8ms/step
Loss: 0.43909767270088196, Accuracy: 0.7746478915214539
```


**NeuralNetwork with Keras tuned hyperparameters**
```python
# Define the deep learning model
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Dense(units=15, activation="tanh", input_dim=21))
nn_model.add(tf.keras.layers.Dense(units=15, activation="tanh"))
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the Sequential model together and customize metrics
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn_model.fit(X_train_scaled, y_resampled, epochs=50)

# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
```python
predictions = nn_model.predict(X_test_scaled)
```
```python
import numpy as np

y_pred = np.round(predictions[:,0])
```
```
Confusion Matrix
   Predicted 0	Predicted 1
Actual 0	1027	189
Actual 1	38	24
Accuracy Score : 0.8223787167449139
Classification Report
              precision    recall  f1-score   support

           0       0.96      0.84      0.90      1216
           1       0.11      0.39      0.17        62

    accuracy                           0.82      1278
   macro avg       0.54      0.62      0.54      1278
weighted avg       0.92      0.82      0.87      1278
```

While all of these models have good accuracy scores a closer examination of the confusion matrix and classification reports reveals that some models were missing many strokes, and also correctly identifying very few. A broad conclusion is that this limited dataset would likely be hard to use for any useful machine learning predictions. The keras tuned neural network had the highest recall out of the models that also achieved higher accuracy scores. The logistic regression model had a much higher recall score than all of the other models at .79, meaning it did the best at not missing any strokes. However, it also had a very large number of incorrect stroke predictions.

In a scenario where the goal is to correctly predict something, a higher precision score might be desirable. However, in this medical context, the goal would be to make sure the model is not incorrectly missing people who were at risk for a stroke, so our logistic regression model did the best for that goal. While not perfect, if the goal is to learn if someone * might * be at a higher risk for stroke based on certain indicators, this model could be useful. Similarly, we might investigate if we could get better results by tweaking parameters in the Decision Tree or Random Forest models.

</details>

## **Outcome**
We were succesfully able to: <br/>
* Train our Machine Learning Model :page_with_curl: <br/> 
* Create various user friendly visuals to support user analysis :chart_with_upwards_trend: <br/>
* Create a Dashboard using Tableau :bar_chart: <br/>
* Create a deployable app as a tangible example of our machine learning model being utilized :100: <br/>

## **Technology/Tools used** :computer:
* Programming Language: Python Pandas, Tableau <br/>
* Packages imported: seaborn, matplotlib.pyplot, pandas, Tensorflow, Scikit-learn, imblearn <br/>
   *From  Scikit-learn: <br/>
      * sklearn.preprocessing: StandardScaler, OneHotEncoder, sklearn.model_selection: train_test_split <br/>
      * sklearn.metrics: accuracy_score, confusion_matrix, classification_report <br/>
      * sklearn.linear_model: LinearRegression, LogisticRegression <br/>
      * sklearn.ensemble: RandomForestClassifier <br/>


## **Team Members:** <br/>
Rina Neaara Bitas: https://github.com/rbitas <br/>
Samuel Fish: https://github.com/samuelhfish <br/>
Philip Lin: https://github.com/PhilipSJLin <br/>
Andrei Tabatchouk: https://github.com/andrei-tabachk <br/>

## **Credits, Copyrights, Resources:** <br/>
* [Giphy](https://giphy.com/) <br/>
* [SlidesCarnival](https://www.slidescarnival.com/) <br/>
* [Google Slides](https://www.google.com/slides/about/) <br/>
* [Kaggle](https://www.kaggle.com/datasets) <br/>

**Note** <br/>
These three websites were used to inform and help use Seaborn plots: <br/>
* https://seaborn.pydata.org/generated/seaborn.histplot.html <br/>
* https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e <br/>
* https://stackoverflow.com/questions/42406233/how-to-add-a-title-to-a-seaborn-boxplot <br/>

