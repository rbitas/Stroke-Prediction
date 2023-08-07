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
Here is the presentation to be accompanied with this project for your viewing pleasure.
   *[Google Slides Presentation](https://docs.google.com/presentation/d/1Y0AWo-qqOz6cqIyUKf_oMHhe51R879zPMlMUvZuqsuw/edit?usp=sharing)

## **Data** :woman_technologist:
For our project, we have visualized data extracted from the following dataset available in the Resources folder <br/>
   * [healthcare-dataset-stroke-data.csv](https://github.com/rbitas/Stroke-Prediction/blob/main/Data/healthcare-dataset-stroke-data.csv) <br/>

## **How to Run**
DATA PREPARATION:

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

```
# Merge one-hot encoded features and drop the originals
df = df.merge(encode_df,left_index=True, right_index=True)
df = df.drop(df_cat,1)
df.head()
```

While we started with a very clean dataset overall, there were still things we needed to adjust including dropping the "id" column which is of no use to us and filling the null values in the bmi column.
```
df = df.drop(columns=['id'])
```

For null bmi values we made the decision to take the mean of the other values and fill the nulls with a number. We decided to use the mean of the all the non-null values.
```
# Find the mean of the "bmi" column to use as replacement for null values.
df["bmi"].mean()
```
```
# Replace null values
df['bmi'] = df['bmi'].fillna(29.88)
df.head()
```
```
v# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

Now we are ready to split the "stroke" column off as our target or "y" variable and turn our remaining columns into our features or the "X" variables.

```
# Remove stroke dtarget from features data
y = df.stroke.values
X = df.drop(columns="stroke").values
```
```
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

ANALYSIS:
After preprocessing our data and testing it in a machine learning model, a 95% accuracy score signaled promising potential for both our data and the initial model. However, a deeper dive into the machine learning reports tells us that is not at all the case. If only 5% of our data set was in the "had stroke" category, the model could predict that no one would have a stroke and achieve 95% accuracy. With this in mind, we realized some kind of oversampling in our model could help with our unbalanced data set.

```python
pip install -U imbalanced-learn
```
```
# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_train, y_train.ravel())
```



Finally, we scaled our data, and then started testing different supervised machine learning and neural network models.

```
# Preprocess numerical data for neural network

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_resampled)
X_test_scaled = X_scaler.transform(X_test)
```
Logistic Regression
```
X_train_scaled.shape
```
```
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='lbfgs',
                                max_iter=200)
classifier
```
```
classifier.fit(X_train_scaled, y_resampled)
```
```
predictions = classifier.predict(X_test_scaled)
results = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)
```
```
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```
```
# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)

# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)
```

```
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
Actual 0	924	292
Actual 1	12	50
Accuracy Score : 0.7621283255086072
Classification Report
              precision    recall  f1-score   support

           0       0.99      0.76      0.86      1216
           1       0.15      0.81      0.25        62

    accuracy                           0.76      1278
   macro avg       0.57      0.78      0.55      1278
weighted avg       0.95      0.76      0.83      1278
```

Random Forest/ Feature Importance Image
```
```
```
```
Decision Tree
```
```
```
```
KerasTuner
```
```
NerualNetwork Keras Tuned
```
```
```
```

While all of these models have good accuracy scores a closer examination of the confusion matrix and classification reports reveals that some models were missing many strokes, and also correctly identifying very few. A broad conclusion is that this limited dataset would likely be hard to use for any useful machine learning predictions. The keras tuned neural network had the highest recall out of the models that also achieved higher accuracy scores. The logistic regression model had a much higher recall score than all of the other models at .79, meaning it did the best at not missing any strokes. However, it also had a very large number of incorrect stroke predictions.

In a scenario where the goal is to correctly predict something, a higher precision score might be desirable. However, in this medical context, the goal would be to make sure the model is not incorrectly missing people who were at risk for a stroke, so our logistic regression model did the best for that goal. While not perfect, if the goal is to learn if someone * might * be at a higher risk for stroke based on certain indicators, this model could be useful. Similarly, we might investigate if we could get better results by tweaking parameters in the Decision Tree or Random Forest models.


## **Outcome**
We were succesfully able to: <br/>
* Train our Machine Learning Model :page_with_curl: <br/> 
* Create various user friendly visuals to support user analysis :chart_with_upwards_trend: <br/>
* Create a Dashboard using Tableau :bar_chart: <br/>

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

