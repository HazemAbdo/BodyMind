# ML Project Document

## Team 11 Members

| Name | Sec | B.N. |
| --- | --- | --- |
| Ahmed Khaled Mahmoud | 1 | 5 |
| Ahmed Sayed | 1 | 6 |
| Ahmed Mahmoud Mohamed | 1 | 12 |
| Hazem Mahmoud Abdo | 1 | 25 |

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

> **Main Evaluation Metric Used is F1Score**
> 

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
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%205.png)

### Features’ Values Distribution

**There's a some observations that we can know from this distribution**

- Most people in dataset with has family history with obesity which makes sense as about half of people in dataset lie in class4
- Most people in dataset eats 3 meals in day ,don’t smoke with low calories burn rate
- More than 25% of people in dataset don’t do Physical activities
- Most people in dataset use Public Transportation and don’t walk

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%206.png)

---

---

## Experiments Results

> Table below shows training and testing scores on different datasets
**- Original dataset without any modification,only encoding categorical values
- Oversampled Dataset using Borderline SMOTE
- Oversampled (ADASYN) Dataset
- Original dataset with BMI feature added
- Oversampled Dataset with BMI feature added
- Oversampled (ADASYN) Dataset with feature BMI added**
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%207.png)

[https://docs.google.com/spreadsheets/d/1eFpWBNf9BuxHH8SHoUv84nRTK-M_48O3reiJv5g4ids/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1eFpWBNf9BuxHH8SHoUv84nRTK-M_48O3reiJv5g4ids/edit?usp=sharing)

### Comments on Experiments Results

> **From results above we can conclude that 
- Bagging Classifier perform better when training on oversampled dataset using adasyn technique with BMI feature added to it with no sign of overfitting
- In general oversampling Improves performance on both models and reduces overfitting
- Both models when trained on original dataset seem to overfit the data**
> 

### Oversampling Techniques Used

- **BorderLine SMOTE**
    - **works by generating synthetic samples along the borderline between the minority and majority classes**.
- **ADASYN**
    - **works by generating synthetic samples for the minority class based on its density distribution. Specifically, it identifies the minority class samples that are harder to classify by the current model and generates synthetic samples by interpolating between these samples and their nearest minority class neighbors.**

## Models

### Logistic Regression

- Params
    - `penalty` This parameter specifies the type of regularization to be applied to the logistic regression model. The two most common types of regularization are L1 regularization and L2 regularization. L1 regularization encourages the model to have sparse feature weights, meaning that some features are given a weight of zero. L2 regularization encourages the model to have small feature weights, meaning that all features are included in the model but with smaller weights.
    - `C` This parameter is the inverse of the regularization strength and controls the amount of regularization applied to the model.
    - `solver`This parameter specifies the algorithm to be used for optimization when fitting the logistic regression model
    - Best Model Params Per each Dataset
        - Original Dataset
            - `LogisticRegression(C=100.0, random_state=42, solver='newton-cg')`
        - Oversampled Dataset
            - `LogisticRegression(C=100.0, random_state=42, solver='newton-cg')`
        - Dataset with feature BMI added
            - `LogisticRegression(random_state=42, solver='newton-cg')`
        - Oversampled dataset with feature BMI added
            - `LogisticRegression(C=100.0, random_state=42, solver='newton-cg')`
    
    ---
    

### Bagging classifier with Decision Tree as base estimator

- Params
    - `n_estimators` This parameter controls the number of base estimators (i.e., decision trees)
     used in the Bagging Classifier. Increasing the number of estimators 
    tends to improve the stability and accuracy of the model, as it reduces 
    the effects of random fluctuations in the training data. However, increasing `n_estimators` also increases the computational complexity of the model
    - `max_samples` This parameter specifies the maximum number of samples to be used in 
    each base estimator. Setting this parameter to a value less than the total number of samples in the training data can help to introduce randomness into the model, which can reduce overfitting and improve generalization performance. However, setting `max_samples` too low can lead to underfitting, as the model may not have enough samples to learn the underlying patterns in the data.
    - `max_features` This parameter specifies the maximum number of features to be considered when splitting a node in a decision tree. Setting this parameter to a 
    value less than the total number of features can help to introduce randomness into the model, which can reduce overfitting and improve generalization performance. However, setting `max_features` too low can lead to underfitting, as the model may not have enough information to make accurate predictions.
- Best Model Params Per each Dataset
    - Original Dataset
        
        `BaggingClassifier(max_features=0.8, max_samples=0.8, n_estimators=50,
                          random_state=42)`
        
    - Oversampled Dataset
        - `BaggingClassifier(max_features=0.7, max_samples=0.6, n_estimators=50,
                          random_state=42)`
    - Dataset with feature BMI added
        - `BaggingClassifier(max_features=0.6, max_samples=0.8, n_estimators=40,
                          random_state=42)`
    - Oversampled dataset with feature BMI added
        - `BaggingClassifier(max_features=0.6, max_samples=0.8, n_estimators=40,
                          random_state=42)`

---

## Regularization & Overfitting Analysis

### Logistic Regression

 

> **Without Oversampling
- Difference between training and cv score is big at low C values and relatively
 decreases as we increase C values, meaning that regularization doesn’t help
overcome overfitting but it seems that it increases diff between cv and train**
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%208.png)

> **With Oversampling (BorderLine SMOTE)
- We noted here that oversampling affects the difference between cv and train scores
 and makes it much smaller but regularization nearly has no effect**
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%209.png)

> **With oversampling (ADASYN)
- Same observations here unless that ADASYN helps in reducing the difference
  between Train and CV scores which is a good sign of overcoming overfitting**
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2010.png)

> Here we noticed that reducing regularization factor improves the scores while using both
L1 and L2 technique, but the overall score is higher while using l2
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2011.png)

### Bagging Classifier

> **Without Oversampling**
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2012.png)

> **With Oversampling (BorderLIne SMOTE)**
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2013.png)

> **With Oversampling (ADASYN)**
> 

![Untitled](ML%20Project%20Document%205fbdd7e091e04ac3b3a32f8f66ab5019/Untitled%2014.png)