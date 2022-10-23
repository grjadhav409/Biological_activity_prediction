
# Doing the necessary imports
from sklearn.model_selection import train_test_split
# from data_ingestion import data_loader
# from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training_Validation_Insertion import train_validation


# Creating the common Logging object


class trainModel():

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self, df):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:

            """ doing the data preprocessing """

            # create separate features and labels
            train_valObj = train_validation(df)
            X, Y = train_valObj.data_cleansing()  # X : morgan fingerprints, y : target ( pchembl )

            # drop prev. scores data
            df = pd.read_csv("results/Scores.csv")
            df1 = df.head(0)
            df2 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
            df2.to_csv("results/Scores.csv", index=False)


            # train test split

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3,
                                                            random_state=36)
            model_finder = tuner.Model_Finder(self.file_object, self.log_writer)  # object initialization

            # getting the best model for each of the clusters
            best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test) # returnd best model name and best model file


            # saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object, self.log_writer)
            file_op.save_model(best_model, best_model_name)

            ## save barplot of scores

            df = pd.read_csv("results/Scores.csv")
            labels = ['SVR                                RF']
            SVR = np.round_(df["SVR"].tolist(), decimals=2)
            RF = np.round_(df["RF"].tolist(), decimals=2)
            x = np.arange(len(labels))  # the label locations
            width = 0.25  # the width of the bars
            fig, ax = plt.subplots(figsize=(10, 12))
            rects1 = ax.bar(x - width / 2, SVR, width, label='SVR')
            rects2 = ax.bar(x + width / 2, RF, width, label='RF')
            ax.set_ylabel('Test R2', fontweight='bold', fontsize=25)
            ax.set_title('Model comparison', fontweight='bold', fontsize=25)
            ax.set_xticks(x, labels, fontsize=25)
            ax.legend()
            ax.bar_label(rects1, padding=3, fontsize=25)
            ax.bar_label(rects2, padding=3, fontsize=25)
            fig.tight_layout()
            plt.savefig("results/scores.png")

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception

