
## Project Goals

In the project, I would like to let the user to explore how different factors, including biological features, other existing symotoms, and behavioural habbits, to better predict heart failure. 


## Design

1) In section 1, I show the overview of the whole dataset by showing the first five rows of data as a dataframe to give user a sense of what does the data look like. I also categorize the data into categorical and quantitative, and let the user to choose whether checking the statistics of quantitative data. The reasons why I split the data into categorical and quantitative is because many further data transformations and analytical approaches are different for those two categories, so I split them in the beginning. 

2) In section 2, I explored the correlation variables among all variables, by splitting variables in categorical and quantitative variables to insepect the correlations among variables. I design the multiselect button to enable user to select which kind of variables to inspect. 

3) In section 3.1, I let the user to choose to visualize the distribution of each variable by splitting variables in categorical and quantitative variables as well. The reason why I didn't choose to let the user to choose every single variable is becasue I think it is better to let the user compare distributions between other variables. 

4) In section 3.2, from the previous visualizations, I found it very interesting that gender seems to be not correlated to the symptoms at all, so I investigate on how does heart failure among gender based on smoking and blood pressure, and anaemia and diabetes. I let user to choose whether he/she wants to further investigate on this.

5) In section 4, based on the statistics/distributions/visualizations I provided before, I let the user to customize what variables seems to be the most important in predicting heart failure, and let user to choose between five popular machine learning models. They would see the accuracies and try different combinations of features and explore different models to validate their intuitions. 



## Develop

I made the whole app by myself, and spending in total about 20 hours in total. The design part is the most difficult: how to make the interaction more fascinating and engaging? 