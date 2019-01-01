# HO_Side_Project





In mobile communication the mobile and base station take the decision to excute the Hand OVER ( HO ) depending on many parameters and many ways
i made simulation to get data to train the model in i useed the following features:

1- average RSRP ( Reference Signal Received Power )  from the serving cell in the last 20 seconds
2- average RSRP ( Reference Signal Received Power ) from the target cell in the last 20 seconds
3- time spent in the serving cell
4- serving cell number
5- target cell number


I saved the data in a csv file after dividing them into train input "x_train.csv" output train "y_train.csv" ( which will be used in train and validation )
and test input data x_test and output test y_test



i made 2 scripts one for training the model and saving the best result " HO_NN.py" it load the csv input and output files
another script for loading the model and test it on the test data " Load_test.py"



it would be better if i got real data which is hard for me now and add other features like data rate




a) to run the project just run the HO_NN.py file keeping all files in the same directory



b) lines from 186 to line 195 are the parameters that can be changed for fine tuning





