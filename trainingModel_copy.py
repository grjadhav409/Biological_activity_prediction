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

    def trainingModel(self, path):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:

            """doing the data preprocessing"""

            # create separate features and labels
            train_valObj = train_validation(path)
            X, Y = train_valObj.data_cleansing()  # X : morgan fingerprints, y : target ( pchembl )



            """ Applying the clustering approach"""

            kmeans = clustering.KMeansClustering(self.file_object, self.log_writer)  # object initialization.
            number_of_clusters = kmeans.elbow_plot(X)  # using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = Y

            # getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            # drop prev. data
            df = pd.read_csv("results/Scores.csv")
            df1 = df.head(0)
            df2 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
            df2.to_csv("results/Scores.csv", index=False)


            for i in list_of_clusters:
                cluster_data = X[X['Cluster'] == i]  # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3,
                                                                    random_state=36)

                # x_train_scaled = preprocessor.standardScalingData(x_train)
                # x_test_scaled = preprocessor.standardScalingData(x_test)

                model_finder = tuner.Model_Finder(self.file_object, self.log_writer)  # object initialization

                # getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)

                # scores.append(scores1xx)
                # index.append(index1)

                # saving the best model to the directory.
                file_op = file_methods.File_Operation(self.file_object, self.log_writer)
                save_model = file_op.save_model(best_model, best_model_name + str(i))

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

            ## save barplot
            df = pd.read_csv("results/Scores.csv")

            labels = ['C0', 'C1', 'C2']
            SVR = np.round_(df["SVR"].tolist(), decimals=2)
            RF = np.round_(df["RF"].tolist(), decimals=2)

            x = np.arange(len(labels))  # the label locations
            width = 0.25  # the width of the bars

            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar(x - width / 2, SVR, width, label='SVR')
            rects2 = ax.bar(x + width / 2, RF, width, label='RF')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Test R2', fontweight='bold', fontsize=15)
            ax.set_xlabel('Cluster', fontweight='bold', fontsize=15)
            ax.set_title('R2 scores of each cluster')
            ax.set_xticks(x, labels)
            ax.legend()
            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)
            fig.tight_layout()
            plt.savefig("results/scores.png")

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception
