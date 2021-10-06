# Interactive Data Science HW2


## Goal Description

This project is tend to help user find out what kind of employee will lead to attrition. In other words, to know if a employee is going to quit their job in advance.

In this project, You'll know the answer of these following questions. You will have more interesting finding when you play around this project.

 - What is the main reason that lead to attrition?
 - What is the relation between different kind of attributes such as age, work over time, monthly income or work life balance?
- Under what circumstance I can get a higher monthly income?  
- Is my employee going to leave my company?

## Rationale of design decisions

 - To answer the first and the second questions, we have to build a correlation coefficient chart to compare each features. I need to use the headmap to achieve this.
 
 - For third question, I build a barchart for specific attribute with respect to others. In this way, we will have better undertanding about what feature relates to others. Also, include a checkbox for user to choose if they want to see the absolute value of the correlation coefficient. (You will know what feature has most significant affect on the feature you choose no matter it is positive correlation or negative correlation)

 - Initially I wanted to put all the barchart on the website, however, I found that I could let user to choose the feature they want to know instead of showing all of them. The user can also discover more interesting thing during their exploration.

- To answer the last question, which is the most interesting part, I need to train the model for this dataset. I choose logistic regression and random forest. I implemented it using scikit-learn since I already have the package for us to easily train the model. The reason I choose logistic regression is because it can show the probabily, I think it's pretty cool!!! And the reason I choose random forest is just because it can get higher accuracy.

- After we get the model, I design a interactive form for you to type your employees' information to predict. 


## Development process

- Built this website myself.
- Spent around 30 to 35 hours.
- I took the most of the time to build those interactive widget and get familiar with streamlit.
- First, I tried to find the dataset on kaggle, and then build the user interface of my website. Thinking about what kind of information I want to show on my website. And then think about different interactive widgets for users to play around. In the last step, train the model and let user type in their information to predict.