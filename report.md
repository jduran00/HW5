## Description of the Project and Problem

The aim of this project is to participate in the Histopathologic Cancer Detection Kaggle competition. I will attempt to create an algorithm that will be able to identify metastatic cancer in images taken from pathology scans. The data for this is provided as files names with an image id. The ground truth is provided in a train_labels.csv and the submission file should also be a .csv file that predicts the probability that the patch in the image contains a pixel or more of tumor tissue. The test and train folders contain .tif files, this is the first time I have encountered this format and I dicovered they store image and graphics. These contain the small image patches of the tissue to be analized. The .csv is the truth for the files in train. The test file is what you use to create the submission file. 

In short, the train folder containing .tif files and the train_labels.csv are a data/results pair and the .tif files in the test folder require you to create their results pair, which will be your submission.csv file. 

To note/cite, this [this](https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy) stack overflow thread was helpful in learning to manage .tif files. 


## Data analysis 

Since I have previously taken a course  that primarily used panadas, that is what I decided to use for this assignment. Using pandas and seaborn it was possible to quickly anazlize the train_labels.csv, I have included a pie chart that demonstrates the split between files with cancer and without cancer in the training dataset. The files provided by kaggle needed little cleaning but to be careful I added a line that would drop rows with missing parts. The provided instructions in Kaggle tells us that there are no duplicates so I did not address them. My plan of anaylsis is use sklearn to train a model and then use that model on the test folder and save the results in a .csv file.


![Figure_1_kaggle.png](attachment:3f15bf35-9ed9-44b3-8532-6c829e5b1346.png)

## Model Architecture

I used a random forest classifier from sklearn.ensemble, due to my previous experience with it and pandas I knew I wanted to use sklearn with it to train a model. I did some research and liked this article [Gradient Boosting vs. Random Forest: Which Ensemble Method Should You Use?](https://medium.com/@hassaanidrees7/gradient-boosting-vs-random-forest-which-ensemble-method-should-you-use-9f2ee294d9c6#:~:text=Two%20of%20the%20most%20popular,operate%20in%20fundamentally%20different%20ways.), I ultimately went with Random Forest because it boasts faster processing and this is a very large data set. 

Random Forest builds many decision trees trained using random subsets of data provided, then combines and averaging the results for the prediction. This is suited to a project like this with binary results.

## Ressults and Anlysis 

## Conclusion 
