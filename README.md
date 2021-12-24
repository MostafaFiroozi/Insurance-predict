# Cross-selling Prediction

```
Mostafa Firoozi Arani
```
## Abstract

In business terminology, Cross-selling refers to offering other products or
services to customers who are purchasing for another product or service. Although
an offer suggestion will not be a great cost for a company, good detection of the
potential customers can reduce the chance of reduction of interest of a customer
due to spam advertisements, and will make the company able to focus more on the
potential customers.

In this project, our aim is to detect potential vehicle insurance buyers, bet`
ween those of whom applied for a health insurance service. There is available a dat
aset with 102351 records, with the columns indicating the data related to each indi
vidual client, as depicted in the table 1. The “Target” attribute indicates that if the s
ubject has been interested in the offer or not.
![image](https://user-images.githubusercontent.com/73081215/146907853-04cf5d6d-4009-4afd-8a83-b9173a4246b5.png)

```

Table 1 Indication of each column data.
```
## Data Preparation

In this phase we investigate our data in terms of homogeneity, existence
of null values, etc. We split the data into categorical and numerical and we will
work on it.
Firstly, we load our dataset in our program after importing the necessary Librar
ies. Till now, the required libs are pandas, seaborn, numpy and mathplotlib.


![image](https://user-images.githubusercontent.com/73081215/146908014-1150ba13-a2db-4e76-97ae-d383c251fe94.png)


Then, a quick glance at our dataset:


![image](https://user-images.githubusercontent.com/73081215/146908070-f1f557e5-0a05-45dd-9820-46f31939afb3.png)

Now, we visualize the how the data is distributed between ‘ 1 ’ and’ 0 ’. Since the gap is not too much, we do not need to down sample the data.

![image](https://user-images.githubusercontent.com/73081215/146908170-03ecdef6-43d3-4387-b92e-ba21e03da1b2.png)
![image](https://user-images.githubusercontent.com/73081215/147357787-17916aa6-46e2-4e81-aa80-3d9ffcc1c624.png)


We use df.info function to study all the info about this dataset. As the belo
w image represents, we can see Data types, Non-Null values. There are some null values in the column “Licence_Type”, which we need to work on.

![image](https://user-images.githubusercontent.com/73081215/146908287-88be41c5-05bb-47ac-adfc-f199fb90c20c.png)


```
Categorical Attributes
```
Using the function” df.isna().sum()”, it is illustrated that there are 5091 null values in that column.

![image](https://user-images.githubusercontent.com/73081215/146908424-3af158b5-4eca-4863-aa36-c9bc00b9cded.png)

We attribute to all the null values in that column, a specific value, “N”, to
be able to monitories the values, aiming to opt for eliminating these ‘null’ values or not.

![image](https://user-images.githubusercontent.com/73081215/146908487-3ba1818c-ac27-4cb6-addc-f32de92fadfb.png)

Now we plot the distribution of all the categorical data between ‘ 0 ’ and’ 1’.
We see in the all licence_Types the values are equally separated between ‘ 0 ’ and ‘ 1 ’, manifesting that it does not have any serious effect on the final prediction. So we can eliminate all the column, with no worry about corruption of the prediction procedure.

![image](https://user-images.githubusercontent.com/73081215/146908662-c6ab2775-5395-4ce2-85f0-41e7e6ea4a11.png)

![image](https://user-images.githubusercontent.com/73081215/146908820-cee24eea-8d4d-45bb-8c74-ac7b2eef3b9e.png)


We also drop the column ‘id’, since it does not have any statistical value for us.

![image](https://user-images.githubusercontent.com/73081215/146908867-69fa0ff7-73f9-49fd-a8f5-295351ab3024.png)


Now we need to define ‘dummy’ values for ‘Categorical’ attributes, to make them
applicable in our ML algorithms.

![image](https://user-images.githubusercontent.com/73081215/146908928-a9b3f1f6-92e2-4fb1-a7c8-73142d4e0bbb.png)

And, Replacing 'objects' with the 'dummy' values.

![image](https://user-images.githubusercontent.com/73081215/146908977-cd2dc97f-77bd-43f5-be49-af5674a8692e.png)

```
Numerical Attributes
```
The very first step with the ‘Numerical’ attributes is to plot the histogram to see its distribution. The below picture represents that, there is a sharp decrease in the ‘Annual_Premium’ attribute. So it would be reasonable the get a logarithm from it using the below code.

![image](https://user-images.githubusercontent.com/73081215/146909059-c491b9af-11e7-456b-ad6f-2cdcea0d847e.png)
![image](https://user-images.githubusercontent.com/73081215/146909292-4ae1a2f4-9aaa-4b43-8314-b0eb106e6a48.png)


We can see in the new histogram that ‘logAnnual_premium’ has a distribution
more similar to normal.

To see how each two ‘Numerical’ attributes distribute in the ‘Target’ value have, we can pair plot all of them a figure. We see the ‘Annual_Premium’ and the ‘log_annual_premium’ have a logarithmic shape relation with each other.



For instance, ‘seniority’ has ‘ gaussian’ distribution, but this distribution is quite the same for ‘ 0 ’ and ‘ 1 ’ which is not a good point. On the other hand, age has two different distribution between ‘ 0 ’ and ‘ 1 ’, make it a more worthy parameter for our prediction.


![image](https://user-images.githubusercontent.com/73081215/146909442-f4bba350-bae7-4b33-85a9-3ba31a44ae89.png)


Now, we will eliminate the ‘Annual_Premium’, since we will be working on
‘logAnnual_Premium’.

Now, we can boxplot the ‘numerical’ attributes to make a comparison between the
different distributions.

![image](https://user-images.githubusercontent.com/73081215/146909566-2d8766fc-aa02-47a4-8023-9aa883dfd0b0.png)


Since we have got distributions so deviated from each others, so we should scale the data to receive similar distributions.

![image](https://user-images.githubusercontent.com/73081215/146909644-7c116e27-98b4-4ae5-a2d5-5a661ff029cd.png)


We define a scaler, feeding the function with our numerical attributes. We will
use this scaler to operate on the other data which we will be using in the future to
predict.

![image](https://user-images.githubusercontent.com/73081215/146909671-ad6bd01b-abe9-4aeb-b04c-a49f36975147.png)



Our scaled dataset will have distributions of the same order for all the ‘numerical’ attributs all arouns ‘ 0 ’.

![image](https://user-images.githubusercontent.com/73081215/146909708-29078ba7-7618-4955-bbb8-ac0a9d134fde.png)


Now we should attach the numerical and categorical attributes, using concat
function. Then dropping ‘Target’, since we want to predict the ‘Target’, not using itfor prediction.

![image](https://user-images.githubusercontent.com/73081215/146909790-1de78dcd-707c-421f-8a61-4150d0602545.png)


**Now our dataset is clear, and ready to work on!!!**


### Sparsing:

We define y as the ‘Target’ Values. Then we seperate train and test parts, considering a 70% to 30% division. We stratify our data, to get the same seperation of ‘ 0 ’ and ‘ 1 ’ both in the train and test set.

![image](https://user-images.githubusercontent.com/73081215/146909920-8867b489-0e2b-416c-8bbe-7d143213d8a3.png)




### Applying PCA

Now we can apply PCA to the data, to check whether we can see a sharp
decrease in the data or not. Unfortunately, there is not, so, applying PCA to this
dataset is not a good idea.

![image](https://user-images.githubusercontent.com/73081215/146910035-40eaf186-64de-4aab-86c4-ca24865a006f.png)


## Prediction

After all this investigations and data preparations, now we can apply our
models to our dataset. First off, we import the necessary libraries.

![image](https://user-images.githubusercontent.com/73081215/146910059-2e805bbf-129f-4f62-8890-bc35f7bf9092.png)


Now we define a function to evaluate the given model with the specified
parameters to find out the best parameters fitting the model. We will compare
different approaches based on f1-score.

![image](https://user-images.githubusercontent.com/73081215/146910081-f8f2a894-dd6f-43ae-a4df-66537d972eb4.png)


Besides, we define another function to plot ROC curve of a specific model,
and calculate the area under that curve.

![image](https://user-images.githubusercontent.com/73081215/146910101-252d80cb-9768-4f3a-8487-4158d07ff2e8.png)


The very first model based on which we want to evaluate is the KNN model. We
import this classifier from sklearn.neighbors library. Then, we define 10 different number of neighbors starting from 10, increasing with 100 steps till 1000. Now we can use our hyper_search function to evaluate this classifier through all this set of parameters. According to our function, the best f-score will be achieved using 110 number of neighbors.

![image](https://user-images.githubusercontent.com/73081215/146910405-aae9677e-ebea-414e-898b-8a9275ffeca0.png)


Then, we plot the ROC curve of the classifier basing on the defined number.

![image](https://user-images.githubusercontent.com/73081215/146910548-6c5864ce-0254-4ed9-b254-dc7e36a5ffce.png)


Although an AUC=084 represents a good behaviuor of the model, we will try other
models to find the best one.


But the resoult of NaiveBase is less promissing, both based on AUC and f 1 - score.

![image](https://user-images.githubusercontent.com/73081215/146910610-109ffc73-64dd-4bd6-8360-7100354e01c4.png)

![image](https://user-images.githubusercontent.com/73081215/146910637-ecc7affd-19b0-4cc4-9c68-2f0c34f1dcd1.png)


It seems the best result will be achieved using RandonForrestClassifier, both in terms of f1-score and AUC.

![image](https://user-images.githubusercontent.com/73081215/146910664-22d1df43-e76b-4251-a53f-72b99ad501bc.png)

Using the offered parameters, we can receive this ROC curve.

![image](https://user-images.githubusercontent.com/73081215/146910679-4e8a9513-b119-41c9-b77a-c24178090c69.png)

We try the Neural-network classifier as well, with 2 different neural architectures, and 4 different learning rates. After trying the best calculated model, we receive ROC curve quite similar to random forest, but since the f1-score in the random forest is better, our final decision would be that.
![image](https://user-images.githubusercontent.com/73081215/146910745-f6756971-eb2a-4acb-b1a7-03042d449e89.png)

![image](https://user-images.githubusercontent.com/73081215/146910855-d1937f4e-ad53-4671-8145-3c59e7634d88.png)

After all, we use all the data to train the model since we do not anymore need to evaluate the data
![image](https://user-images.githubusercontent.com/73081215/146910878-c106dbc8-982d-4e65-95a7-770393954fd8.png)



