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

**9.** One of the first steps of exploratory data analysis should always be to look at what **the values of y look like**. We typically do this by checking the distribution of y: a visual description of what the possible values of y can be and how often they occur (histogram).

**10.** The **head of a distribution is a range where there are many values**. A **long tail** of a distribution is a case when many values are spread very far from the head — and these values visually appear as a “tail” on the histogram.

**11.**  The **long tail makes it quite difficult for us to see the distribution, but it has an even stronger effect on a model: such distribution can greatly confuse the model, so it won’t learn well enough**. So as a part of exploratory data analysis, we need to check the distribution of the target variable. If the target distribution has a long tail, we need to apply the log transformation. Without it, we may get inaccurate and misleading predictions from the linear regression model.

One way to solve this problem is **log transformation**. If we apply the log function to the prices, in our example, it removes the undesired effect

**y_new = log (y + 1)** we can achieve this log transformation through a numpy function **np.log1p** like 
        
        y_new = np.log1p(y)

The +1 part is important in cases that have zeros. The logarithm of zero is minus infinity, but the logarithm of one is zero. If our values are all non-negative, by adding 1, we make sure that the transformed values do not go below zero.

**np.log1p** :

np.log1p() is a NumPy function that returns the **natural logarithm of 1 + x**. It is useful when x is very small, because it allows for more accurate computation than directly computing np.log(x + 1). This is because when x is very small, the 1 can be lost in the floating point representation of the number, leading to loss of precision.

**np.expm1**:

To get back the original value of x after applying np.log1p(), you can use the np.expm1() function. **np.expm1() is the inverse of np.log1p()**, and computes exp(x) - 1, which is the reverse of the operation performed by np.log1p(x).


**12.** Generally, it’s good when the **target distribution looks like the normal distribution**. Under this condition, models such as linear regression perform well.

**13. Normal equation:**

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

+ it is only for linear regression and does not generalize to other learning algorithms 
+ solve for w and b without iteration
+ normal equation method may be used in machine learning libraries that implement linear regression (maybe useful tip in a job interview) 
+ normal equation is slow when number of features is large (> 10000)
+ keep in mind that GD is the recommended method for finding parameters w and b

**14.** When I have categorical data I want to apply one hot encoding, I do not need to have all the categories, within one column I mean, which may be even a bad thing and leading to an error in model training! What I learned that was to go with the like most 5 categories like I can find the most common ones easily through using value_counts like

    df['market_category'].value_counts().head()

so what I did was to perform one hot encoding using my following function

    def category_maker(df, col_name_list):
        return pd.get_dummies(df, columns=col_name_list)  

then I only will choose the one hot coded columns craeted based on the most common categories like I first need to find their names like: 

    market_category_cat = list(df['market_category'].value_counts().head().index)

    added_market_category_categorical = [] 
    for one_item in market_category_cat:
        added_market_category_categorical.append('market_category_'+ one_item)

then 

    encoded_data = category_maker (df, (['market_category']))
    encoded_data_final = encoded_data.loc[:, added_market_category_categorical]

**15. Regularization**: We saw that adding new features does not always help, and in our case, it made things a lot worse. The reason for this behavior is **numerical instability**. One of the terms in the normal equation is the **inverse of the (np.dot(X.T, X)) matrix**. The **inversion is the issue** in our case. Sometimes, when adding new columns to X, we can accidentally add a column that is a combination of other columns. For example, if we already have the mile per gallon (MPG) in the city feature and decide to add kilometers per liter in the city, the second feature is the same as the first one but multiplied by a constant. When this happens, np.dot(X.T, X) becomes **undetermined or singular, which means that it’s not possible to find an inverse for this matrix**. If we try to invert a singular matrix, NumPy will tell us about that by raising a **LinAlgError: Singular matrix**.

We may not get this error but end up having a bad model with high rmse, in these cases our code doesn’t raise any exceptions. It happened because we don’t typically have columns that are perfect linear combinations of other columns. The real data is often **noisy**, with measurement errors (such as recording 1.3 instead of 13 for MPG), rounding errors (such as storing 0.0999999 instead of 0.1), and many other errors. Technically, such matrices are not singular, so NumPy doesn’t complain.

**For this numerical instability, some of the values in the weights become extremely large — a lot larger than they are supposed to be**. In numerical linear algebra, such issues are called numerical instability issues, and they are typically solved with **regularization techniques**. The **aim of regularization is to make sure that the inverse exists by forcing the matrix to be invertible**. Regularization is an important concept in machine learning: it means **“controlling” — controlling the
weights of the model so that they behaive correctly and don’t grow too large**.

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

**16.** Feature scaling in applying GD vs. Normal equation

from chatGPT:
In general, you **don't need to perform feature scaling when using the normal equation for linear regression. However, feature scaling can sometimes help to improve the numerical stability of the computations.**

**On the other hand, when using gradient descent, it is highly recommended to perform feature scaling to help the algorithm converge faster and more reliably.** This is because gradient descent works better when the features are on a similar scale. When the features are on different scales, the optimization algorithm might take longer to converge, or it might oscillate back and forth, making it difficult to find the optimal solution.

**17.** scikit-learn implementation of LinearRegression = Normal eq. + regularization

my question for chatGPT:

does **from sklearn.linear_model import LinearRegression** use normal equation?

Re:

Yes, the LinearRegression model from scikit-learn does use the **normal equation** to solve for the parameters of the linear regression model.

The normal equation is a closed-form solution to the linear regression problem, which involves finding the values of the model coefficients that minimize the sum of squared errors between the predicted values and the actual values in the training data.

**The scikit-learn implementation of LinearRegression uses a variant of the normal equation that includes regularization to prevent overfitting of the model.** Specifically, scikit-learn uses the **Ordinary Least Squares (OLS) method, which is a common method for solving the normal equation.**



<a name="3"></a>
## 3. Machine learning for classification

**18.** Logistic regression is a workhorse of machine learning and the most widely used model in the industry.

**19.** How to identify which features are important in our data? Through implement feature importance.

One way to identify which features are important in your data is to implement feature importance analysis. Feature importance is a technique used in machine learning to determine the relative importance of each feature or variable in predicting the target variable.

There are different methods to calculate feature importance, such as:

+ Recursive Feature Elimination (RFE): RFE is a backward feature selection method that recursively removes features and builds a model on the remaining features until a desired number of features is reached. The importance of each feature is determined by the order in which they are removed.

+ Random Forest Feature Importance: Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy. The importance of each feature is determined by measuring how much the performance of the model decreases when the feature is randomly shuffled or removed.

+ Gradient Boosting Feature Importance: Gradient Boosting is a machine learning technique that builds an ensemble of weak prediction models in a stepwise manner. The importance of each feature is determined by measuring how much the loss function of the model increases when the feature is removed.

By implementing feature importance analysis, we can identify which features have the most significant impact on the target variable and use them to build more accurate predictive models. This can also help in reducing the dimensionality of the data, which can improve the performance and interpretability of the model.

**20.** a couple of good tricks when doing EDA:

1. Try to transpose the df to see all the columns in the screen like

    df.head().T

2. Watch out for cases when you expect a column to be numeric, but Pandas says it’s not: most likely the column contains special encoding (like a space " " ) for missing values that require additional preprocessing.

We can force this column to be numeric by converting it to numbers using a special function in Pandas: **to_numeric**. By default, this function raises an exception when it sees nonnumeric data (such as spaces), but we can make it skip these cases by specifying the **errors='coerce' option. This way Pandas will replace all nonnumeric values with a NaN (not a number)** like: 

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['TotalCharges'] = df['TotalCharges'].fillna(0)

I achived this using

    df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '0').astype(float)

3. Another good thing to check is to see if there is any space in the values of the columns with object type and if so replace it with let's say underscores:

    string_columns_list = df_2.dtypes[df_2.dtypes =='object'].index
    
    for one_col in string_columns_list:
    
        df_2[one_col] = df_2[one_col].str.lower().str.replace(' ', '_')

4. I need to take care of categorial target variables in a way that yes/no wil be converted to 1/0:

        df_2['Churn'] = (df_2['Churn'] == 'yes').astype(int)

**21.** How to deal with imbalanced data?

The data set could be either imbalanced or balanced, in a balanced dataset the **positive and negative classes are equally distributed among all observations.** So before going ahead and train the model first figure out if the dataset is whether or not balanced. 

Suppose you are working on a project where the data is imbalanced. What techniques would you use to address this problem?

chatGPT response:

Imbalanced data is a common problem in machine learning where one class has significantly fewer samples than the other class(es). In such cases, the classifier may be biased towards the majority class, leading to poor performance on the minority class. Here are some techniques that can be used to address imbalanced data:

+ Resampling techniques: Resampling techniques involve either oversampling the minority class or undersampling the majority class to balance the dataset. Oversampling techniques include duplicating the minority class samples or generating new synthetic samples using techniques like SMOTE (Synthetic Minority Over-sampling Technique). Undersampling techniques involve randomly removing samples from the majority class to match the number of samples in the minority class.

+ Cost-sensitive learning: Cost-sensitive learning involves assigning different misclassification costs to different classes based on their importance. The misclassification cost of the minority class is generally higher than the majority class, and the classifier is penalized more for misclassifying minority class samples.

+ Ensemble methods: Ensemble methods like bagging and boosting can be used to improve the performance of classifiers on imbalanced data. These methods involve combining multiple classifiers trained on different subsets of the data to improve the overall performance.

Random Forest and XGBoost are both ensemble methods that can be used to address imbalanced data problems in classification tasks.

Random Forest is a bagging ensemble method that trains multiple decision trees on random subsets of the training data and aggregates their predictions to make the final classification decision. The randomization of the training data subsets and the selection of a random subset of features for each tree helps to reduce overfitting and improve the generalization performance of the classifier. In the case of imbalanced data, Random Forest can help to balance the classification performance by considering different subsets of the minority class during training and improving the overall classification performance.

XGBoost is a boosting ensemble method that combines weak learners (typically decision trees) to create a stronger classifier. XGBoost iteratively trains decision trees on the residual errors of the previous iteration, focusing on the samples that were misclassified in the previous iteration. XGBoost also uses regularization techniques such as L1 and L2 regularization, which helps to reduce overfitting and improve the generalization performance of the classifier. In the case of imbalanced data, XGBoost can help to balance the classification performance by giving more weight to the minority class during training and improving the overall classification performance.

Both Random Forest and XGBoost have been shown to be effective in handling imbalanced data problems in classification tasks. However, the choice between the two methods may depend on the specific characteristics of the dataset and the problem at hand. In general, Random Forest may be more suitable for datasets with moderate imbalances, while XGBoost may be more suitable for datasets with severe imbalances.

+ Threshold adjustment: The decision threshold of the classifier can be adjusted to increase the recall of the minority class. This involves changing the threshold at which a sample is classified as positive or negative to increase the sensitivity towards the minority class.

+ Use of evaluation metrics: Evaluation metrics like precision, recall, F1 score, and AUC ROC can be used to evaluate the performance of the classifier on imbalanced data. These metrics can provide a better understanding of the classifier's performance on the minority class and help in selecting the appropriate technique to address the imbalanced data problem.

It is important to note that the choice of technique depends on the specific problem and the dataset. It is also important to ensure that the technique does not introduce bias or overfitting to the data. Therefore, a careful analysis of the data and the problem is necessary before selecting a technique to address the imbalanced data problem.

so this imballanced dataset issue is only for classification and not regression right?

chatGPT response:

Imbalanced data is typically considered an issue in classification problems, where the goal is to predict a categorical label for each input sample. In classification problems, imbalanced data can result in a classifier that is biased towards the majority class, leading to poor performance on the minority class.

In regression problems, where the goal is to predict a continuous output value for each input sample, imbalanced data is not usually a concern. However, there can be other issues related to the distribution of the output values, such as skewness or outliers, which can affect the performance of the regression model. These issues may require different techniques to address, such as data normalization or the use of robust regression methods.

That being said, there can be cases in which imbalanced data may also occur in regression problems. For example, if the range of the output values is large and the majority of the samples fall within a narrow range, the regression model may be biased towards predicting values within that range, leading to poor performance on samples outside of that range. In such cases, techniques such as data normalization or the use of different loss functions may be necessary to address the imbalanced distribution of the output values.

**22.** feature importance analysis 

Knowing how other variables affect the target variable, churn, is the key to understanding the data and building a good model. This process is called feature importance analysis, and it’s often done as a part of exploratory data analysis to figure out which variables will be useful for the model. It also gives us additional insights about the dataset and helps answer questions like “What makes customers churn?” and “What are the characteristics of people who churn?” We have two different kinds of features: **categorical and numerical**. Each kind has different ways of measuring feature importance, so we will look at each separately:

**22.1. feature importance analysis - categorical features**

**1.** Comparing the group rate and the global rate:
We can look at all the distinct values of a variable. Then, for each variable, there’s a group of customers: all the customers who have this value. For each such group, we can compute the churn rate, which is the group churn rate. When we have it, we can compare it with the global churn rate — the churn rate calculated for all the observations at once.

If the **difference** between the rates, the group rate vs. global rate, is small, the value is not important when predicting churn because this group of customers is not really different from the rest of the customers. On the other hand, if the difference is not small, something inside that group sets it apart from the rest. A machine learning algorithm should be able to pick this up and use it when making predictions.

**2.** In addition to looking at the difference between the group rate and the global rate, it’s interesting to look at the **ratio** between them. In statistics, the ratio between probabilities in different groups is called the **risk ratio**, where risk refers to the risk of having the effect. In our case, the effect is churn, so it’s the risk of churning:

    risk = group rate / global rate

Risk is a number **between zero and infinity**. It has a nice interpretation that tells you how likely the elements of the group are to have the effect (churn) compared with the entire population.

If the difference between the group rate and the global rate is small, the risk is close to 1: this group has the same level of risk as the rest of the population. Customers in the group are as likely to churn as anyone else. In other words, a group with a risk close to 1 is not risky at all. 

If the risk is lower than 1, the group has lower risks: the churn rate in this group is smaller than the global churn. For example, the value 0.5 means that the clients in this group are two times less likely to churn than clients in general. On the other hand, if the value is higher than 1, the group is risky: there’s more churn in the group than in the population. So a risk of 2 means that customers from the group are two times more likely to churn.

here is the code to explore this:

    from IPython.display import display
    df_final_group = pd.DataFrame()
    for one_cat_col in cat_columns:
        df_group = df_2.groupby(one_cat_col)['Churn'].agg(['mean'])
        df_group['diff'] = df_group['mean'] -  df_2['Churn'].value_counts(normalize= True)[1]
        df_group['risk'] = df_group['mean'] / df_2['Churn'].value_counts(normalize= True)[1]
        display(df_group) 

**feature importance analysis - categorical features -- Mutual Information**

The kinds of differences we just explored are useful for our analysis and important for understanding the data, but it’s hard to use them to say what the most important feature is and whether each feature is more useful than the other.

Luckily, the **metrics of importance** can help us: we can measure the **degree of dependency between a categorical variable and the target variable**. If two variables are dependent, knowing the value of one variable gives us some information about another. The higher the degree of dependency, the more useful a feature is. On the other hand, if a variable is completely independent of the target variable, it’s not useful and can be safely removed from the dataset.

For **categorical variables, one such metric is mutual information**, which tells how much information we learn about one variable if we learn the value of the other variable. It’s a concept from information theory, and in machine learning, we often use it to measure the mutual dependency between two variables.

Higher values of mutual information mean a higher degree of dependence: if the mutual information between a categorical variable and the target is high, this categorical variable will be quite useful for predicting the target. On the other hand, if the mutual information is low, the categorical variable and the target are independent, and thus the variable will not be useful for predicting the target. Mutual information is already implemented in Scikit-learn in the mutual_info_ score function from the metrics package, so we can just use it:

    from sklearn.metrics import mutual_info_score
    
    mi_dic = {}
    for one_cat_col in cat_columns:
        mi_dic[one_cat_col] = mutual_info_score (df_2[one_cat_col], df_2['Churn'])

    pd.DataFrame.from_dict(mi_dic, orient='index', columns=['MI']).sort_values('MI', ascending = False)

**22.2. feature importance analysis - numerical features -- (Pearson’s) correlation coefficient**

Mutual information is a way to quantify the degree of dependency between **two categorical variables**, but it **doesn’t work when one of the features is numerical**. We can, however, measure the dependency between a binary target variable and a numerical variable. We can pretend that the **binary variable is numerical (containing only the numbers zero and one) and then use the classical methods from statistics to check for any dependency between these variables**.One such method is the **correlation coefficient (sometimes referred as Pearson’s correlation coefficient)**. It is a value from –1 to 1:

+ Positive correlation means that when one variable goes up, the other variable tends to go up as well. In the case of a binary target, when the values of the variable are high, we see ones more often than zeros. But when the values of the variable are low, zeros become more frequent than ones.

+ Zero correlation means no relationship between two variables: they are completely independent.

+ Negative correlation occurs when one variable goes up and the other goes down. In the binary case, if the values are high, we see more zeros than ones in the target variable. When the values are low, we see more ones.

here is my code for calculating Pearson’s correlation coefficient:

    numerical_cols = ['Monthlycharges' , 'Totalcharges', 'Tenure']
    corelation_coef_dic = {}

    for one_col in numerical_cols:
        corelation_coef_dic[one_col] = df_2[one_col].corr(df_2['Churn'])

    pd.DataFrame.from_dict(corelation_coef_dic,  orient='index', columns=['CC']).sort_values(by = 'CC', ascending = False)

shorter implementation of my code above:

    pd.DataFrame(df_2[numerical_cols].corrwith(df_2['Churn']), columns = ['CC']).sort_values(by = 'CC', ascending = False)

**23.** feature importance analysis should be done **ONLY** on training data so I need to split data first into training, val, and test to make sure no knowledge can be leaked from test dataset

My question for chatGPT:
should i perform feature importance analysis before or after splitting data into training, val, and testing? and why? 

Re:
It is generally recommended to perform **feature importance analysis after splitting your data into training, validation, and testing sets.**

The main reason for this is to **avoid any data leakage, which can occur if you perform feature importance analysis on your entire dataset before splitting it.** Data leakage can happen when information from the validation or test set inadvertently leaks into the training set, leading to overly optimistic performance estimates and poor generalization to new data.

By splitting your data before performing feature importance analysis, **you can ensure that you are only using information from the training set to select important features**, and that **your validation and test sets remain untouched until you are ready to evaluate your model's performance on unseen data.** This helps to **prevent any potential biases or overfitting that could arise from using all of your data at once.**

In summary, it's best to split your data into training, validation, and testing sets before performing feature importance analysis to ensure that you are following best practices for machine learning and avoiding any data leakage issues.

**23.1** I actually did not use the info we gained from feature importance analysis like i expected to perform some kind of feature selection based on that but we didnot!!! yes in the end Alexey mentioned this and suggested to redo the exercise with applying this knowledge!:)

my question for chatGPT:
what is the whole point of feature importance analysis? we do it to perform feature selection, right? i mean we included only the important features in the model training instead of all the features?

Re:
Yes, the primary purpose of feature importance analysis is to identify the most important features for a given machine learning problem. Once we have identified the most important features, we can use this information to perform feature selection, which involves selecting a subset of the features that are most relevant to the target variable and excluding the rest.

Feature selection is important because it can help improve the performance of a machine learning model by reducing overfitting, increasing interpretability, and reducing training and inference time. By removing irrelevant or redundant features, we can simplify the model and improve its generalization performance on new, unseen data.

There are different techniques for feature selection, including **filter methods, wrapper methods, and embedded methods**. **Feature importance analysis is typically used as a filter method, which means that it ranks the importance of each feature based on some criterion and selects the top-ranked features for inclusion in the model**. Common criteria for ranking feature importance include the magnitude of the feature coefficients in a linear model, the decrease in impurity in a decision tree-based model, or the permutation importance in a tree-based or ensemble model.

It's important to note that feature importance analysis is not always necessary or appropriate for every machine learning problem. For example, some models like neural networks are less interpretable and may not provide feature importance scores. In other cases, the data may have low feature dimensionality or the features may all be important, making feature selection less necessary. Ultimately, the decision to perform feature importance analysis and feature selection depends on the specific machine learning problem and the characteristics of the data.

**24. feature engineering -- encoding**

**After performing feature importance analysis**, and **before we proceed to training**, however, we need to perform the feature engineering step: **transforming all categorical variables to numeric features**. 

We cannot just take a categorical variable and put it into a machine learning model. The **models can deal only with numbers in matrices**. So, we need to convert our categorical data into a matrix form, or encode. **One such encoding technique is one-hot encoding.**

We can use Scikit-learn to perform one-hot encoding. We can perform one-hot encoding in multiple ways in Scikit-learn, but we will use **DictVectorizer.** As the name suggests, DictVectorizer takes in a dictionary and vectorizes it — that is, it creates vectors from it. Then the vectors are put together as rows of one matrix.

**side note**: To convert a dataframe to a list of dictionaries, we use the **to_dict** method with the **orient="records"** parameter:

    train_dict = df_train[categorical + numerical].to_dict(orient='records')

Here is the code to perform one-hot encoding using DictVectorizer (I use OneHotEncoder from sklearn though):

    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    from sklearn.feature_extraction import DictVectorizer
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)
    X_train = dv.transform(train_dict)
    dv.get_feature_names()

The DictVectorizer class can take in a set of parameters. We specify one of them: **sparse=False**. This parameter means that the created matrix will not be sparse and instead will create a simple NumPy array.

**24.1** I need to encode the new datapoint in the same way as I encoded my dataset used in modeling, so here is how i can save the encoder object to be able to use it later

**Save the encoder object to be used later**

    import pickle
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
Preprocess the new data using the same encoder object

    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    encoded_new_data = encoder.transform(new_data[cat_features])

**25. Logistic regression**

**side note:** Linear regression is a linear model. It’s called linear because it **combines the weights of the model with the feature vector linearly, using the dot product.** Linear models are simple to implement, train, and use. Because of their simplicity, they are also fast.

Logistic regression is also a linear model, but unlike linear regression, it’s a classification model, not regression, even though the name might suggest that. It’s a **binary classification model, so the target variable yi is binary; the only values it can have are zero and one.** Observations with yi = 1 are typically called positive examples: examples in which the effect we want to predict is present. Likewise, examples with yi = 0 are called negative examples: the effect we want to predict is absent.

The **output of logistic regression is probability — the probability that the observation xi is positive, or, in other words, the probability that yi = 1.**

To be able to treat the output as a probability, we need to make sure that the predictions of the model always stay between zero and one. We use a special mathematical function for this purpose called **sigmoid**, and the full formula for the logistic regression model is

![](https://raw.githubusercontent.com/DanialArab/images/main/ML_bookcamp/logistic_regression_formula.PNG)

If we compare it with the linear regression formula, the only difference is this sigmoid function: in case of linear regression, we have only w0 + xiTw. This is why both of these models are linear; they are both based on the dot product operation.

The sigmoid function maps any value to a number between zero and one. It’s defined this way:

![](https://raw.githubusercontent.com/DanialArab/images/main/ML_bookcamp/sigmoid.PNG)

![](https://raw.githubusercontent.com/DanialArab/images/main/ML_bookcamp/sigmoid_function.PNG)

The sigmoid function outputs values that are always between 0 and 1. When  the input is 0, the result of sigmoid is 0.5; for negative values, the results are below 0.5 and start approaching 0 for input values less than **–6**. When the input is positive, the result of sigmoid is above 0.5 and approaches 1 for input values starting from **6.**

The parameters of the logistic regression model are the same as for linear regression:

    w0 is the bias term.
    w = (w1, w2, …, wn) is the weights vector
    

**How to code logistic regression**

    model = LogisticRegression(solver='liblinear', random_state=1)

random_state. The seed number for the random-number generator. It shuffles the data when training the model; to make sure the shuffle is the same every time, we fix the seed.

Other useful parameters for the model include C, which controls the regularization level. Specifying C is optional; by default, it gets the value 1.0.

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)

The result of predict_proba is a two-dimensional NumPy array, or a two-column matrix. The first column of the array contains the probability that the target is negative (no churn), and the second column contains the probability that the target is positive (churn). These columns convey the same information. We know the probability of churn — it’s p — and the probability of not churning is always 1 – p, so we don’t need both columns.

    y_pred_soft = model.predict_proba(X_val)[:, 1]

This output (probabilities) is often called **soft predictions**. These tell us the probability of churning as a number between zero and one. It’s up to us to decide how to interpret this number and how to use it.

To make the actual decision about whether to send a promotional letter to our customers, using the probability alone is not enough. We need **hard predictions — binary values of True (churn, so send the mail) or False (not churn, so don’t send the mail).**

    y_pred_hard = (y_pred > threshold).astype(int)

    accuracy = (y_val == y_pred_hard).mean()

**side_note:**
model.predict **(y_pred = model.predict(X_val))** does not have such a parameter to specify threshold?

That's correct, the predict method in scikit-learn does not have a parameter to specify the threshold for classification. The predict method simply returns the predicted class labels for the input data, based on the **default threshold of 0.5 for binary classification** (and argmax for multiclass classification).

If you want to adjust the classification threshold for your model, you will need to use the **predict_proba method** instead, which returns the predicted probabilities of each class. You can then apply your own threshold to these probabilities to make the final classification decision.

**How to get the model parameters**

    bias_term = model.intercept_[0]

The rest of the weights are stored in **model.coef_[0]**. To see which feature is associated with each weight: 

    feature_names = list(X.columns)

get the absolute values of the coefficients

    coef_abs = np.abs(model.coef_)

calculate the feature importance scores

    feature_importance = np.sum(coef_abs, axis=0)

    feat_imp_dict = dict(zip(feature_names, feature_importance))

    for feature, importance in sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True):
        print(feature + ': {:.2f}'.format(importance))
        
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
