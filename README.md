# ML Project Document

## Structure
- Inference Code located at Test Code Folder,including the model used
- Notebook that contains the project source code
- Link to Colab Notebook that contains the same content of the notebook


## Project Description:

**The main objective of this project is to apply machine learning concepts and algorithms to a real-world
problem. The selected problem for this semester is “Body Level Classification”. First, you need to explore
and analyze the given dataset to find the best way to approach it. Then, you have to apply at least three
machine learning algorithms (taught in this course) to the dataset in the aim of solving the problem.
Moreover, you are advised to explore further methods to enhance the performance.**

---

## Problem and Dataset Description

**You are required to solve a classification problem for human body level based on some given attributes
related to the physical, genetic and habitual conditions. The given attributes are both categorical and
continuous. The human body level can be categorized into (4 levels/classes).
You are given 16 attributes and 1477 data samples, where classes are not evenly distributed. Try to build
models that can adapt to the class imbalance to achieve the best possible results.**

> **Main Evaluation Metrics Printed are Accuracy and weighted F1**

---

## Dataset Analysis Results

### Distribution of Body Levels Classes

```cpp
The Below Graph shows that the dataset classes is imbalanced in favor of
class 'Body Level 4' with 46%
```

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled.png)

### Explore Relationship between weight and height

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%201.png)

### Box Plot Of each feature

```cpp
This box plot of features shows that nearly all features have no outliers except
for age columns with multiple outliers and weight column.
So let's explore Age column in more details
```

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%202.png)

### Explore Age Column

```cpp
This distribution shows that ~55% of the people in the dataset lie in the range
from 20-30 years.
```

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%203.png)

### Covariance Matrix

### Covariance Matrix shows that

- Weight has a very high variance and covariance with other variables, particularly with Age and Fam_Hist. This suggests that weight may be an important predictor of many health outcomes and may be strongly related to age and family history
- Age has a relatively high variance and covariance with many other variables, suggesting that age may be an important predictor of many health outcomes
- Height has a low variance and covariance with other variables, suggesting that it may not be a very important predictor of health outcomes.
- H_Cal_Consump has a relatively low covariance with other variables, suggesting that it may not be strongly related to other aspects of health **which seems strange.**
- Phys_Act has a relatively high covariance with Meal_Count, suggesting that physical activity may be related to eating habits
- Phys_Act has a relatively high negative covariance with Weight, suggesting that physical activity may be related to losing weights

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%204.png)

### Mean Weight Of each class

> Following our observation that the Weight has high covariance with a lot of features, now we see its mean within different classes which push us towards that it has huge effect in classifying body level

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%205.png)

### BMI Range for each Class

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%206.png)

### Features’ Values Distribution

**There's a some observations that we can know from this distribution**

- Most people in dataset with has family history with obesity which makes sense as about half of people in dataset lie in class4
- Most people in dataset eats 3 meals in day ,don’t smoke with low calories burn rate
- More than 25% of people in dataset don’t do Physical activities
- Most people in dataset use Public Transportation and don’t walk

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%207.png)

---

---

## Experiments Results

> Table below shows training and testing scores on different datasets
> \*\*- Original dataset without any modification,only encoding categorical values

- Oversampled Dataset using Borderline SMOTE
- Oversampled (ADASYN) Dataset
- Original dataset with BMI feature added
- Oversampled Dataset with BMI feature added
- Oversampled (ADASYN) Dataset with feature BMI added\*\*
  >

### Original Dataset

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%208.png)

### BLSMOTE Dataset

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%209.png)

### ADASYN Dataset

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2010.png)

### Original Dataset BMI

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2011.png)

### BLSMOTE with BMI

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2012.png)

### ADASYN with BMI

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2013.png)

### Comments on Experiments Results

- **Adding the BMI feature to the original data improves the performance of all models. This suggests that the BMI feature contains valuable information that is relevant to the classification task.**
- **Oversampling the original dataset using BLSMOTE leads to mixed results. While it improves the performance of the logistic regression and SVC models, it decreases the performance of the bagging classifier. This suggests that the effectiveness of BLSMOTE may depend on the specific machine learning model used.**
- **Oversampling the original dataset using ADASYN leads to improved performance for all models. This suggests that ADASYN is a more robust oversampling technique that can be effective across different machine learning models.**
- **Oversampling the dataset that has BMI feature using BLSMOTE leads to mixed results, similar to oversampling the original dataset. This suggests that the effectiveness of BLSMOTE may depend on the presence of certain features in the dataset.**
- **Oversampling the dataset that has BMI feature using ADASYN leads to improved performance for all models, similar to oversampling the original dataset with ADASYN. This suggests that ADASYN is a more robust oversampling technique that can be effective even when new features are added to the dataset.**

---

### Oversampling Techniques Used

- **BorderLine SMOTE**
  - **works by generating synthetic samples along the borderline between the minority and majority classes**.
- **ADASYN**
  - **works by generating synthetic samples for the minority class based on its density distribution. Specifically, it identifies the minority class samples that are harder to classify by the current model and generates synthetic samples by interpolating between these samples and their nearest minority class neighbors.**
  ***

## Models

### Logistic Regression

- Params
  - `Penalty type (L1 or L2)`The penalty type determines the type of regularization applied to the model. L1 regularization (Lasso) applies a penalty on the absolute value of the coefficients, which can result in sparse models where some coefficients are set to zero. L2 regularization (Ridge) applies a penalty on the squared value of the coefficients, which can result in a smoother model. L1 regularization is useful for feature selection, while L2 regularization is useful for reducing the impact of outliers.
  - `Regularization parameter (C)` This hyperparameter controls the amount of regularization applied to the model. If C is small, the model will be more heavily regularized, which can help prevent overfitting by reducing the complexity of the model. On the other hand, if C is large, the model will be less regularized, which can result in higher model complexity
  - `solver`This parameter specifies the algorithm to be used for optimization when fitting the logistic regression model
  ***

### Bagging classifier with Decision Tree as base estimator

- **Params**
  - ```````Num of estimators` This parameter controls the number of base estimators (in our case decision trees) used in the Bagging Classifier. Increasing the number of estimators
    tends to improve the stability and accuracy of the model and reduce the variance of the model, as it reduces the effects of random fluctuations in the training data. However, increasing `n_estimators` also increases the computational complexity of the model
  - `Max Samples` This parameter specifies the maximum number of samples to be used in each base estimator. Setting a lower value for this feature (can be considered some sort of regularization) can help to introduce randomness into the model and reduce model complexity, which can reduce overfitting and improve generalization performance. However, setting `max_samples` too low can lead to underfitting, as the model may not have enough samples to learn the underlying patterns in the data.
  - `Max Features` This parameter specifies the maximum number of features to be considered when splitting a node in a decision tree. Setting this parameter to a
    value less than the total number of features can help to introduce randomness into the model, which can reduce overfitting and improve generalization performance. However, setting `max_features` too low can lead to underfitting, as the model may not have enough information to make accurate predictions.

### SVM

- **Params**
  - `Kernel type` SVM can use different kernel functions to transform the input data into a higher-dimensional space where it can be separated by a hyperplane. The most common kernel types are linear, polynomial, radial basis function (RBF), and sigmoid. The choice of kernel type can affect the performance of the model. Linear kernel is useful for linearly separable data, while non-linear kernels like RBF and polynomial can be used for non-linearly separable data.
  - `Regularization parameter (C)` This hyperparameter controls the trade-off between maximizing the margin and minimizing the classification error. If C is small, the model will be more heavily regularized, which can help prevent overfitting by reducing the complexity of the model. On the other hand, if C is large, the model will be less regularized, which can result in higher model complexity.
  - **`Degree (for polynomial kernel only)`** This hyperparameter determines the degree of the polynomial function used by the polynomial kernel. A higher degree can result in a more complex decision boundary, while a lower degree can result in underfitting.

---

## Hyperparameters Analysis

### Logistic Regression

> \*\*Without Oversampling

- The difference between the training and cross-validation (CV) scores is small for low values of the regularization parameter
  C, but increases as the value of C increases. This suggests that the model with regularization does not overfit the data, but has a high bias
  due to the regularization penalty. As the regularization effect is decreased by increasing the value of C, the model becomes more
  flexible and is able to fit the training data more closely, resulting in a decrease in bias and an increase in the overall training and CV
  scores. However, as the model becomes more flexible, it also becomes more prone to overfitting the data, which is reflected in the increasing
  difference between the training and CV scores. Therefore, the choice of the regularization parameter C involves a trade-off between bias and variance, and it is important to select a value that balances these two sources of error.\*\*
  >

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2014.png)

> \*\*With Oversampling (BorderLine SMOTE)

- We noted here that oversampling affects the difference between cv and train scores
  and makes it much smaller but regularization nearly has no effect\*\*
  >

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2015.png)

> \*\*With oversampling (ADASYN)

- at low values of C, the model has low training,cv scores and low differences between them
- as we increase C, the training,cv scores increases but with relatively high difference between them
- at higher values of C, the difference between training,cv decreseas again which is a good sign of overcoming overfitting\*\*
  >

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2016.png)

> \*\*Original Data with BMI

- Same observations as original Data but with relatively more difference between cv,train scores\*\*
  >

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2017.png)

> **BLSMOTE Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2018.png)

> **ADASYN Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2019.png)

> Here we noticed that reducing regularization factor improves the scores while using both
> L1 and L2 technique, but the overall score is higher while using l2

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2020.png)

### Bagging Classifier

> **Without Oversampling**

- The graph of n_estimators versus the train and CV scores for a bagging classifier shows that increasing the number of estimators generally improves the
  overall performance of the model. Specifically, as n_estimators increases, both the train and CV scores tend to increase, indicating
  that the model is becoming more accurate and better able to capture the underlying patterns in the data.The decrease in the difference between the train and CV scores as
  n_estimators increases suggests that the model is becoming less prone to overfitting but by comparing with graphs below it has some signs of overfitting. This is because bagging is a technique that reduces overfitting by aggregating predictions from multiple independently trained models. As the number of estimators increases, the variance of the ensemble decreases, which in turn reduces the difference between the train and CV scores.
- As the value of max_samples increases, the number of samples used to train each base estimator also increases, which in turn reduces the variance of the ensemble and makes the model less prone to overfitting but by comparing with graphs below it has some signs of overfitting
- increasing "max_features" leads to a model with greater complexity, which is able to capture more information from the training data and
  therefore perform better on both the training and validation sets but it still suffers from overfitting.

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2021.png)

> **With Oversampling (BorderLIne SMOTE)**

- **Same comments unless that oversampling the dataset using BLSMOTE increases the validation scores which indicate that the model learns better from oversampled dataset compared to original one**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2022.png)

> **With Oversampling (ADASYN)**

- **Same comments unless that**
  - **oversampling the dataset using ADASYN increases the validation scores which indicate that the model learns better from oversampled dataset compared to original one**
  - **The validation scores starts to decrease as max_features exceeds .7 which means that the model has been more complex to the extent that it overfitting the training data and perform worse
    on cross validation data**
  - **Overall difference between training and cross validation scores has been very small which means that the model nearly overcome overfitting problem.**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2023.png)

> **Original Data with BMI**

- **Adding BMI Feature to the dataset helps the model a lot to learn and increasing the cross validation scores on avg to .99 with very low difference with training score**
- **The validation scores starts to decrease as max_features exceeds .7 which means that the model has been more complex to the extent that it overfitting the training data and perform worse**
- **The validation scores reaches its maximum in n_estimator graph at .4, then starts to decrease again and increases the difference with training score**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2024.png)

> **BLSMOTE with BMI**

- **Same comments but with on avg higher cv score ~.997**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2025.png)

> **ADASYN with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2026.png)

> **Bagging Classifier (effect of number of estimators on Training VS CV) on the same plot**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2027.png)

### SVC

> **Without Oversampling**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2028.png)

> **With Oversampling (BorderLIne SMOTE)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2029.png)

> **With Oversampling (ADASYN)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2030.png)

> **Original Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2031.png)

> **BLSMOTE with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2032.png)

> **ADASYN with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2033.png)

---

---

## Learning Curve

### Logistic Regression

- **Comments**
  - **The learning curves for the logistic regression model trained on the original data and the oversampled data using BLSMOTE and ADASYN, with and without BMI, suggest that the performance of the model is highly dependent on the dataset's characteristics and the sampling method used.**
  - **For the model trained on the original data without modification, the learning curve shows that as the size of the training data increases, the model's train score gradually decreases while the cross-validation score steadily increases. This suggests that the model is suffering from overfitting and requires more diverse training data to generalize better to unseen data.**
  - **When oversampling the original data using BLSMOTE, the learning curve shows a rapid increase in the train score as the training data size increases. The cross-validation score also steadily increases, suggesting that the model is generalizing well to unseen data. However, the initial cross-validation score is lower than that of the model trained on the original data without modification, indicating that the oversampled data may not fully capture the distribution of the original data.**
  - **Using ADASYN for oversampling the original data, the learning curve also shows a rapid increase in the train score as the training data size increases. The cross-validation score is higher than that of BLSMOTE, suggesting that ADASYN produces more diverse samples.**
  - **When using BMI as an additional feature in the original data, the learning curve shows that the model can achieve a higher cross-validation score than without BMI. However, the model's train score slightly decreases as the training data size increases, suggesting that the model is not overfitting but may be underfitting with a small training size.**
  - **When oversampling the data with BMI using BLSMOTE, the learning curve shows a slight increase in the train score as the training data size increases. The cross-validation score decreases slightly, indicating that the oversampled data may not fully capture the distribution of the original data.**
  - **Using ADASYN for oversampling data with BMI, the learning curve shows that the model can achieve a high cross-validation score. The train score slightly decreases as the training data size increases, suggesting that the model is not overfitting but may be underfitting with a small training size.**
  - **Overall, the results suggest that oversampling techniques can improve the performance of the logistic regression model on imbalanced data. ADASYN produces more diverse samples than BLSMOTE, leading to better performance. Including BMI as an additional feature can also improve the model's performance**

> **Without Oversampling**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2034.png)

> **With Oversampling (BorderLine SMOTE)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2035.png)

> **With oversampling (ADASYN)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2036.png)

> **Original Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2037.png)

> **BLSMOTE Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2038.png)

> **ADASYN Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2039.png)

### Bagging Classifier

- **Starting with the original data, the model has a high training score of 100 and a cross-validation score of 85, indicating that the model is overfitting to the training data and not generalizing well to new data. However, as the training size increases, the model's performance on the cross-validation set improves, reaching a score of 96.9. This suggests that the original dataset may have been insufficiently large to train the model effectively and this difference between training and cross-validation suggests that the model may be overfitting.**
- **When oversampling the original data using BLSMOTE or ADASYN, the model's training score remains at 100, but the cross-validation score starts lower at 70 or 74, respectively. As the training size increases, the cross-validation score improves, reaching a maximum of 98 for BLSMOTE and 99.1 for ADASYN. This suggests that oversampling the data can help improve the model's performance, particularly when using ADASYN.**
- **Adding a new feature, BMI, to the original dataset improves the model's cross-validation score to 98.5, indicating that this feature is informative and helps the model better generalize to new data. However, as the training size increases, the cross-validation score increases slightly to 99.3, suggesting that the model is becoming more stable and genralizable.**
- **When oversampling the data that includes BMI using BLSMOTE or ADASYN, the model's training score remains at 100, but the cross-validation score starts lower at 87. As the training size increases, the cross-validation score improves slightly to 98 for BLSMOTE and to 100 for ADASYN. This suggests that oversampling the data can still improve the model's performance.**
- **Overall, these learning curves demonstrate the importance of having a sufficiently large and diverse dataset for training machine learning models. Oversampling techniques such as BLSMOTE and ADASYN can help improve performance when the dataset is imbalanced, but the choice of oversampling technique can also affect the degree of improvement. Additionally, incorporating informative features can help the model better generalize to new data.**

> **Without Oversampling**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2040.png)

> **With Oversampling (BorderLIne SMOTE)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2041.png)

> **With Oversampling (ADASYN)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2042.png)

> **Original Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2043.png)

> **BLSMOTE with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2044.png)

> **ADASYN with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2045.png)

### SVC

- **\*Starting with the original data, the model has a high training score of 100 and a cross-validation score of 82, indicating that the model is overfitting to the training data and not generalizing well to new data. However, as the training size increases, the model's performance on the cross-validation set decreases slightly, reaching a score of 99.4 and cross-validation score increases to reach 97.8 meaning that
  the model is becoming more genralizable but with some room to improvement \*\***
- **When oversampling the original data using BLSMOTE or ADASYN, the model's training score remains at 100, but the cross-validation score starts lower at 70 or 63, respectively. As the training size increases, the cross-validation score improves, reaching a maximum of 98.9 for BLSMOTE and 99.4 for ADASYN. This suggests that oversampling the data can help improve the model's performance, particularly when using ADASYN.**
- **Adding a new feature, BMI, to the original dataset improves the model's cross-validation score to 94.5, indicating that this feature is informative and helps the model better generalize to new data. However, as the training size increases, the cross-validation score increases slightly to 98.7, suggesting that the model becomes more genralize**
- **When oversampling the data that includes BMI using BLSMOTE or ADASYN, the model's training score remains at 100, but the cross-validation score starts lower at 59,67 respectively. As the training size increases, the cross-validation score improves slightly to 99.6 for BLSMOTE and to 99.7 for ADASYN. This suggests that oversampling the data can still improve the model's performance.**
- **Overall, these learning curves demonstrate the importance of having a sufficiently large and diverse dataset for training machine learning models. Oversampling techniques such as BLSMOTE and ADASYN can help improve performance when the dataset is imbalanced, but the choice of oversampling technique can also affect the degree of improvement. Additionally, incorporating informative features can help the model better generalize to new data.**

> **Without Oversampling**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2046.png)

> **With Oversampling (BorderLine SMOTE)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2047.png)

> **With oversampling (ADASYN)**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2048.png)

> **Original Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2049.png)

> **BLSMOTE Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2050.png)

> **ADASYN Data with BMI**

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2051.png)

---

## Bias Variance Decomposition

### Bagging Classifier with Data contained BMI oversampled using ADASYN

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2052.png)

---

# conclusion

- **Most models suffer from overfitting when trained on original data.**
- **Adding BMI Feature makes the model underfit at small training sizes.**
- **Adding BMI Feature which is more correlated feature with target variable helps the model to learn better the true function and improves the overall performance of all models.**
- **Oversampling Using ADASYN technique ,on average, affects the performance of the models more than BLSOMTE technique.**
- **Oversampling and affecting performance depend much on the model type.**

---
