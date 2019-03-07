# petfinder kaggle submission
## by Robin Brecht and Philipp Paraguya

This is a submission project for the petfinder.my kaggle challenge:

https://www.kaggle.com/c/petfinder-adoption-prediction

### General ideas

Idea | Description | Status
--- | --- | ---
Democratic Ensembles | When ensembling the models you can go for multiple strategies. For Example: democratic, uniform, median | Not implemented
Optimized Rounding | Classic roundings use .5 as threshold. But maybe when rounding the classification with kappa as scorer another threshold is needed | Implemented
Regression | This problem could also be solved using a regression model instead of classification. In combination with optimized rounding it could improve the accuracy. Maybe even ensemble the different types after | Not implemented

### Ideads for Features

Feature | Description | Fields to use
--- | --- | ---
Name/NoName | Maybe there is a relation between pets who already have names and pets who don't have a name | Name
Name length | Does the length of a name influence the decision to adopt? | Name
Cute Names | With online naming databases as a basis, can we categorize names as "cute" or rather "more adoptable"? | Name
Image quality| Maybe there is a relation between the quality of the images and the adoption time | ???
Length of description|Does the amount of information about the pet influence the decsision to adopt?| calculated field
sentiment of description|  Does the sentiment of the description influence the decision to adopt?|???