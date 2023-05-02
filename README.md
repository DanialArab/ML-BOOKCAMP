# ML-BOOKCAMP

This repository documents my notes from the book Machine Learning Bookcamp by Alexey Grigorev. The jupyter notebook files containing my solutions to the book's projects are attached to this repo. 

# Table of content

1. [Intro to ML](#1)

2. [Machine learning for regression](#2)

3. [Machine learning for classification](#3)

4. [Evaluation metrics for classification](#4)

5. [Deploying machine learning models](#5)

6. [Decision trees and ensemble learning](#6)

7. [Neural networks and deep learning](#7)

8. [Serverless deep learning](#8)

9. [Serving models with Kubernetes and Kubeflow](#9)


<a name="1"></a>
## 1. Intro to ML

This section contains very basic stuff, which was a good review and so I took some notes, as follow:

**1.** Machine learning projects are different from traditional software engineering projects, which are **(rule-based solutions)**

**2.** **CRISP-DM** is a step-by-step methodology for implementing successful machine learning projects

**3.** In summary, the difference between a traditional software system and a system based on machine learning:

In machine learning, we give the system the input and output data, and the result is a model (code) that can transform the input into the output. The difficult work is done by the machine; we need only supervise the training process to make sure that the model is good. In contrast, in traditional systems, we first find the patterns in the data ourselves and then write code that converts the data to the desired outcome, using the manually discovered patterns. So we first need to come up with some logic: a set of rules for converting the input data to the desired output. Then we explicitly encode these rules using a programming language such as Java or Python, and the result is called software. So, in contrast with machine learning, a human does all the difficult work.

**4.** different types of supervised learning algorithms:

+ regression

+ classification (binary vs. multiclass)

+ **ranking**: the **target variable y is an ordering of elements within a group**, such as the order of pages in a search-result page. The problem of ranking often happens in areas like **search and recommendations**, but it’s out of the scope of this book and we won’t cover it in detail.

**5.** **Cross-Industry Standard Process for Data Mining (CRISP-DM)**

Certain techniques and frameworks help us organize a machine learning project in such a way that it doesn’t get out of control. One such framework is CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining. It was invented quite long ago, in 1996, but in spite of its age, it’s still applicable to today’s problems. According to CRISP-DM (figure 1), the machine learning process has six steps:

        1 Business understanding
        2 Data understanding
        3 Data preparation
        4 Modeling
        5 Evaluation
        6 Deployment
        
![alt text](https://raw.githubusercontent.com/DanialArab/images/main/ML_bookcamp/CRISP-DM_process.PNG)

**Figure 1 The CRISP-DM process. A machine learning project starts with understanding the problem and then moves into data preparation, training the model, and evaluating the results. Finally, the model goes to deployment. This process is iterative, and at each step, it’s possible to go back to the previous one.**

**6.** Testing a model on a live system is called **online testing**, and it’s important for evaluating the quality of a model on real data. This approach, however, belongs to the evaluation and deployment steps of the process, not to the modeling step.

**7.** If we repeat the process of model evaluation over and over again and use the **same validation dataset** for that purpose, the good numbers we observe in the validation dataset may appear just by chance. In other words, the “best” model may simply get lucky in predicting the outcomes for this particular dataset. In statistics and other fields, this problem is known as the **multiple-comparisons problem or multiple-tests problem**. The more times we make predictions on the same dataset, the more likely we are to see good performance by chance. To guard against this problem, we use the same idea: we hold out part of the data again. We call this part of data the test dataset. We use it rarely, only for testing the model that we selected as the best.

**8.** It’s important to use the model selection process and to validate and test the models in **offline settings** first to make sure that the models we train are good. If the model behaves well offline, we can decide to move to the next step and deploy the model to evaluate its performance with real users.


<a name="2"></a>
## 2. Machine learning for regression

# 9. 
One of the first steps of exploratory data analysis should always be to look at what **the values of y look like**. We typically do this by checking the distribution of y: a visual description of what the possible values of y can be and how often they occur. This type of visualization is called a histogram.

# 10. 
The **head of a distribution is a range where there are many values**. A **long tail** of a distribution is a case when many values are spread very far from the head — and these values visually appear as a “tail” on the histogram.

# 11. 
The **long tail makes it quite difficult for us to see the distribution, but it has an even stronger effect on a model: such distribution can greatly confuse the model, so it won’t learn well enough**. So as a part of exploratory data analysis, we need to check the distribution of the target variable. If the target distribution has a long tail, we need to apply the log transformation. Without it, we may get inaccurate and misleading predictions
from the linear regression model.

One way to solve this problem is **log transformation**. If we apply the log function to the prices, in our example, it removes the undesired effect

**y_new = log (y + 1)** we can achieve this log transformation through a numpy function **np.log1p** like y_new = np.log1p(y)

The +1 part is important in cases that have zeros. The logarithm of zero is minus infinity, but the logarithm of one is zero. If our values are all non-negative, by adding 1, we make sure that the transformed values do not go below zero.

**np.log1p** :

np.log1p() is a NumPy function that returns the **natural logarithm of 1 + x**. It is useful when x is very small, because it allows for more accurate computation than directly computing np.log(x + 1). This is because when x is very small, the 1 can be lost in the floating point representation of the number, leading to loss of precision.

**np.expm1**:

To get back the original value of x after applying np.log1p(), you can use the np.expm1() function. np.expm1() is the inverse of np.log1p(), and computes exp(x) - 1, which is the reverse of the operation performed by np.log1p(x).


# 12. 
Generally, it’s good when the **target distribution looks like the normal distribution**. Under this condition, models such as linear regression perform well.


# 13. Normal equation:

I can get the model's weights using normal equation:

![](https://raw.githubusercontent.com/DanialArab/images/main/ML_bookcamp/normal_eq.PNG)


here is how I **coded normal equation**:

    def dummy_added (X):
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])

    def normal_eq(X, y):
        X = dummy_added(X)
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        
    w = normal_eq (X_train, y_train)
    
    def prediction (X, w):
        return np.dot(X, w)

    y_pred = prediction (dummy_added(X_test) , w)
    
    sns.histplot(y_pred, label='prediction', bins = 60, color = 'black')
    sns.histplot(y_train, label='target', bins =60)
    plt.legend()


**note from Andrew's course on normal eq.:** 

* it is only for linear regression and does not generalize to other learning algorithms 
* solve for w and b without iteration
* normal equation method may be used in machine learning libraries that implement linear regression (maybe useful tip in a job interview) 
* normal equation is slow when number of features is large (> 10000)
* keep in mind that GD is the recommended method for finding parameters w and b

# 14. 
When I have categorical data I want to apply one hot encoding, I do not need to have all the categories, within one column I mean, which may be even a bad thing and leading to an error in model training! What I learned that was to go with the like most 5 categories like I can find the most common ones easily through using value_counts like

**df['market_category'].value_counts().head()**

so what I did was to perform one hot encoding using like my following function

    def category_maker(df, col_name_list):
        return pd.get_dummies(df, columns=col_name_list)  

then I only will choose the one hot coded columns craeted based on the most common categories like I first need to find their name like: 

    market_category_cat = list(df['market_category'].value_counts().head().index)

    added_market_category_categorical = [] 
    for one_item in market_category_cat:
        added_market_category_categorical.append('market_category_'+ one_item)

then 

    encoded_data = category_maker (df, (['market_category']))
    encoded_data_final = encoded_data.loc[:, added_market_category_categorical]

# 15. Regularization

We saw that adding new features does not always help, and in our case, it made things a lot worse. The reason for this behavior is **numerical instability**. One of the terms in the normal equation is the **inverse of the (np.dot(X.T, X)) matrix**. The **inversion is the issue** in our case. Sometimes, when adding new columns to X, we can accidentally add a column that is a combination of other columns. For example, if we already have the mile per gallon (MPG) in the city feature and decide to add kilometers per liter in the city, the second feature is the same as the first one but multiplied by a constant. When this happens, np.dot(X.T, X) becomes **undetermined or singular, which means that it’s not possible to find an inverse for this matrix**. If we try to invert a singular matrix, NumPy will tell us about that by raising a **LinAlgError: Singular matrix**.

We may not get this error but end up having a bad model with high rmse, in these cases our code doesn’t raise any exceptions. It happened because we don’t typically have columns that are perfect linear combinations of other columns. The real data is often **noisy**, with measurement errors (such as recording 1.3 instead of 13 for MPG), rounding errors (such as storing 0.0999999 instead of 0.1), and many other errors. Technically, such matrices are not singular, so NumPy doesn’t complain.

**For this numerical instability, some of the values in the weights become extremely large — a lot larger than they are supposed to be**. In numerical linear algebra, such issues are called numerical instability issues, and they are typically solved with **regularization techniques**. The **aim of regularization is to make sure that the inverse exists by forcing the matrix to be invertible**. Regularization is an important concept in machine learning: it means **“controlling” — controlling the
weights of the model so that they behave correctly and don’t grow too large**.

One way to do regularization is to add a small number to each diagonal element of the matrix. Then we get the following formula for linear regression:

![](https://raw.githubusercontent.com/DanialArab/images/main/ML_bookcamp/regularized_lienar_regression.PNG)

Here alpha controls the amount of regularization. Regularized linear regression is often called **ridge regression**. Many libraries, including Scikit-learn, use ridge to refer to regularized linear regression and linear regression to refer to the unregularized model.

Regularization affects the final solution by making the components of w smaller. We can see that the more regularization we add, the smaller the weights become.

note: Sometimes, when adding a new feature causes performance degradation, simply removing this feature may be enough to solve the problem. Having a **validation dataset** is important to decide whether to add regularization, remove the feature, or do both: we use the score on the validation data to choose the best option. 

here is my code to perform ridge (normalized linear regression - normal equation):

    def regularized_normal_eq(X, y, alpha):
        X = dummy_added(X)
        reg_term = np.eye(np.dot(X.T, X).shape[0]) * alpha 
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + reg_term), X.T), y)

# 16. Feature scaling in applying GD vs. Normal equation

from chatGPT:
In general, you **don't need to perform feature scaling when using the normal equation for linear regression. However, feature scaling can sometimes help to improve the numerical stability of the computations.**

**On the other hand, when using gradient descent, it is highly recommended to perform feature scaling to help the algorithm converge faster and more reliably.** This is because gradient descent works better when the features are on a similar scale. When the features are on different scales, the optimization algorithm might take longer to converge, or it might oscillate back and forth, making it difficult to find the optimal solution.

# 17. scikit-learn implementation of LinearRegression = Normal eq. + regularization
my question for chatGPT:

does **from sklearn.linear_model import LinearRegression** use normal equation?

Re:

Yes, the LinearRegression model from scikit-learn does use the **normal equation** to solve for the parameters of the linear regression model.

The normal equation is a closed-form solution to the linear regression problem, which involves finding the values of the model coefficients that minimize the sum of squared errors between the predicted values and the actual values in the training data.

**The scikit-learn implementation of LinearRegression uses a variant of the normal equation that includes regularization to prevent overfitting of the model.** Specifically, scikit-learn uses the **Ordinary Least Squares (OLS) method, which is a common method for solving the normal equation.**




<a name="3"></a>
## 3. Machine learning for classification

<a name="4"></a>
## 4. Evaluation metrics for classification

<a name="5"></a>
## 5. Deploying machine learning models

<a name="6"></a>
## 6. Decision trees and ensemble learning

<a name="7"></a>
## 7. Neural networks and deep learning

<a name="8"></a>
## 8. Serverless deep learning

<a name="9"></a>
## 9. Serving models with Kubernetes and Kubeflow
