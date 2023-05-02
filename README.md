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

This section contains basic stuff, which was a good review and so I took some notes, as follow:

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
