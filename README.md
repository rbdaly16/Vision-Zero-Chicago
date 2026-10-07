# **The Road to Vision Zero**

## **Authors**:
Robert Daly, DS; Michael Romanski, DS; Nicholas Tjandra, DS

![image](https://github.com/nickthetj/DSC-Phase3-Project/assets/126971652/9e88f1d6-2353-4ad1-95ec-1f7329edad13)


## **Overview**
The City of Chicago would like to gain more insight into traffic accidents given information about the car, the people in the car, and the road conditions. They plan on pursuing the goals of Vision Zero, an organization with a strategy to eliminate all traffic fatalities and severe injuries, while increasing safe, healthy, equitable mobility for all. Our project involves the examination of crash data sourced from the City of Chicago's Data Portal to further this mission. By analyzing this data, we hope to create a predictive model that can help evaluate trends across crashes. Through this comprehensive analysis, we will be able to make recommendations for Chicago to make their streets safer. 


## **Business Problem**
Vision Zero believes that the loss of even a single life is unacceptable. This is especially true because traffic accidents are not actually accidents at all; they are crashes, because they are preventable. In order to move Chicago closer to this vision, our data analysis led to the following recommendations:

Pedestrian Safety<br>
Traffic Rule Adherence<br>

Continue below to see how we arrived at these recommendations and what Chicago can do to prioritize these measures.

## **Data Understanding**
![image](https://github.com/nickthetj/DSC-Phase3-Project/assets/126971652/5bb378f0-e0b8-467f-bb04-e8fbb7767fa5)

The datasets used in this analysis come from the City of Chicago's Data Portal. We analyzed over 740,00 crashes from the City of Chicago's Data Portal ranging from 2016 to 2022. This data contained information on the crashes themselves, the outcome of the people involved, and information on the vehicles involved. Due to the size of this data, it was unable to be stored in GitHub. To access the data, please click on the links here:<br>
Crashes: https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if <br>
People: https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d <br>
Vehicles: https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3 <br>

## **Data Preparation and Analysis**
A core component of Vision Zero is that the loss of even a single life is unacceptable. To gain insight into how to prevent deaths, we classified the data into two categories: crashes that resulted in Fatalities, and crashes that did not result in Fatalities. 

We trained a Logistic Regression Model as well as a Decision Tree Model. As Fatal crashes only consisted of about 1% of our dataset, we used a Pipeline to supplement the class with SMOTE to allow our model to better learn this category. We one hot encoded all 14 categorical columns which allowed the model to learn to classify each crash as Fatal or Not Fatal based on 238 features such as the weather, the time of day, the type of crash, and more. 

We tuned our model with GridSearch to constantly improve our recall score. We chose to focus on recall score to minimize the likelihood of our model predicting a crash as Not Fatal, when it was in fact Fatal.

## **Model**
Our best model was the Logistic Regression with a recall score of .70. Aa you can see in the confusion matrix, the model is minimizing the amount of False Negatives. In other words, the model is min!
imizing the chance that it predicts a crash as Not Fatal when it was in fact Fatal. <br>
![Screenshot 2023-08-25 at 4 27 25 PM](https://github.com/nickthetj/DSC-Phase3-Project/assets/126971652/f10271c2-7444-4899-8d63-2be27491937f)

Our model also revealed which coefficients were most relevant in terms of predicting fatalities. Here you can see that dark blue is the Primary Cause of Crash, light blue is the Traffic Control Device, orange is the Crash Type, and yellow is the Maneuver the car took resulting in the crash. <br>
![Sheet 2 (1)](https://github.com/nickthetj/DSC-Phase3-Project/assets/126971652/bfa16c00-66b8-498e-bea6-6f0a592fddd9)

To get a further analysis on those coefficients, we calculated the Fatality percentage for each. <br>
![Sheet 3 (3)](https://github.com/nickthetj/DSC-Phase3-Project/assets/126971652/8c68050a-9466-4418-bf19-91270fcbb117) <br>
To no surprise, the top cause of a fatal crash is the Physical Condition of the Driver. Following that is Exceeding Authorized Speed Limit and Driving on the wrong side of the road. While Physical Condition of the Driver is a pressing issue, we came up with an easy to implement recommendation that may be able to curb the speed limit issue and Driving on the Wrong side of the road. As for the Crash Type, we focused our recommendation on minimizing Pedestrian fatalities, but believe it can also minimize the other features as well. 

## **Recommendations**
We recommend Chicago look to the Strong Towns organization and what they have accomplished in restructuring the development of cities to be more suitable and safer for citizens.

Our first recommendation is to redesign your crosswalks. Raising the Crosswalks increases its visibility and makes it clear to both driver and pedestrian that this area of the street is designed for the pedestrian's safety first by forcing the driver to slow down as they approach.<br>
![image](https://github.com/nickthetj/DSC-Phase3-Project/assets/126971652/7f05dd49-1770-48a6-8b30-e2e964a71a86)

Our Second recommendation is to narrow the size of the street by introducing bike lanes. Many city streets have low speed limits but are wide by design so that drivers feel comfortable speeding. By tightening the space around them and introducing bake lanes with barriers to separate them from the the car lanes, drivers will be more attentive to their surroundings and more inclined to slow down.<br>
![image](https://github.com/nickthetj/DSC-Phase3-Project/assets/126971652/2ae7b332-5d82-42c8-8aff-b9999b83ae10)

In conclusion, our model with .70 recall score shows two main ideas: <br>
One, that Pedestrians and Cyclists are most at risk for fatalities.<br>
And two, that fatal crashes are likely to be caused by drivers not following the rules of the road.<br>
We believe Chicago can move closer to zero fatalities by implementing raised crosswalks and narrower streets via protected bike lanes.

## **Next Steps**
There are a few steps we would like to take to further limit fatalities in car crashes:<br>
1. Crash Location Investigation
2. BAC Investigation
3. Driverless Cars Investigation
   
For the crash location investigation, the datasets we worked with provided some information available on the location of every crash, but it was broken down by street. It might be easier and more efficient to break the city down into areas or districts to highlight particular problem areas of the city that could be reformed.

For the BAC investigation, the top preliminary cause and fatal percentage of fatal crashes was the Physical Condition of the Driver. With more time we could investigate how much of this was due to drunk drivers or drivers under the influence of drugs.

For the Driverless Cars Investigation, we would like to investigate the Driverless feature more thoroughly and see how the rise in popularity of self-driving vehicles has impacted this.

## **Thank You**
Thank you for taking the time to review our recommendations.
We hope this information helps and we look forward to working with you more on the next steps.

Sincerely, <br>
Robert Daly, Michael Romanski, Nicholas Tjandra <br>

## Further Details
Further details are available in the full analysis presented in the [Jupyter Notebook](https://github.com/nickthetj/DSC-Phase3-Project/blob/main/Final_EDA_Notebook.ipynb). <br>
See our presentation [here!](https://www.youtube.com/watch?si=nQMl9WNnnF5rCmVa&t=1157&v=0UXYrxgU7dE&feature=youtu.be)

## Repository Structure
```
├── data
├── notebooks
├── README.md
├── LICENSE
├── .gitignore
├── .DS_Store
├── A Predictive Analysis of Fatal Traffic Crashes in Chicago.pdf
└── Final EDA Notebook.ipynb
```
