## Description of the Project and Problem

The aim of this project is to participate in the Histopathologic Cancer Detection Kaggle competition. I will attempt to create an algorithm that will be able to identify metastatic cancer in images taken from pathology scans. The data for this is provided as files names with an image id. The ground truth is provided in a train_labels.csv and the submission file should also be a .csv file that predicts the probability that the patch in the image contains a pixel or more of tumor tissue. The test and train folders contain .tif files, this is the first time I have encountered this format and I dicovered they store image and graphics. These contain the small image patches of the tissue to be analized. The .csv is the truth for the files in train. The test file is what you use to create the submission file. 

In short, the train folder containing .tif files and the train_labels.csv are a data/results pair and the .tif files in the test folder require you to create their results pair, which will be your submission.csv file. 

To note/cite, this [this](https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy) stack overflow thread was helpful in learning to manage .tif files. 


## Data analysis 

Since I have previously taken a course  that primarily used panadas, that is what I decided to use for this assignment. Using pandas and seaborn it was possible to quickly anazlize the train_labels.csv, I have included a pie chart that demonstrates the split between files with cancer and without cancer in the training dataset. The files provided by kaggle needed little cleaning but to be careful I added a line that would drop rows with missing parts. The provided instructions in Kaggle tells us that there are no duplicates so I did not address them. My plan of anaylsis is use sklearn to train a model and then use that model on the test folder and save the results in a .csv file.

![Figure_1_kaggle](https://github.com/user-attachments/assets/b7518f4f-0ded-4f5a-bdf1-a341490ff986)


## Model Architecture

I used a random forest classifier from sklearn.ensemble, due to my previous experience with it and pandas I knew I wanted to use sklearn with it to train a model. I did some research and liked this article [Gradient Boosting vs. Random Forest: Which Ensemble Method Should You Use?](https://medium.com/@hassaanidrees7/gradient-boosting-vs-random-forest-which-ensemble-method-should-you-use-9f2ee294d9c6#:~:text=Two%20of%20the%20most%20popular,operate%20in%20fundamentally%20different%20ways.), I ultimately went with Random Forest because it boasts faster processing and this is a very large data set. 

Random Forest builds many decision trees trained using random subsets of data provided, then combines and averaging the results for the prediction. This is suited to a project like this with binary results.

## Results and Analysis 
I have pasted a screenshot of my submission below. 
<img width="950" alt="kagglesubmission" src="https://github.com/user-attachments/assets/8a51a364-9302-4329-864d-4919969dcb63" />

This could be way higher as show in other submissions. I go over ideas on how to fix this below. 

Another way of checking my results I added a submission_analysis.py file that would create side by side charts and comparasions of the train_model.csv and my submissions.csv . I also tested my model by running it on the train files and then testing the accuracy on the ground truth values. The result was about 72%. I know this could be better and have a couple ideas as to how. 


My number one issue is that I only used 1000 samples to create the model instead of the 220026 provided. I know this was counterintutive and no the project but everytime more than 3000 samples was used every app and tab open on my laptop would turn black and shutdown, including the code editor I was using. Ideally, I would find a solution but I wished to submit what I currentlly had done within the due date. Another issue I ran into was a memory error, I did not have enough spcae on my laptop. I implmented batch testing and kept it in groups of 1000 and then I added those to a list of predictions before moving on to the next step. This prevented the error. 

Outside of those intial issues, another way I could improve my results is to use a more accurate model. The random forest classifier works but spesifically with a large data set like this one, and after looking at top preforming submissions on kaggle I think a neural network could improve the results. 


## Conclusion 

In conclusion, my approach taught me a lot and allowed me to learn common issues with classifying a large dataset. I see ways I could make this better and hope to spend more time learning from other approches on Kaggle and by contiuing to learn about AI and ML. 
