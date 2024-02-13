from check_submission import compare_files
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

"""
As stated in the report.pdf this python file is only used for our personal tryouts
And is NOT used in the code of the assignment
"""

def main(true_file, pred_file):
    '''
    Prints the accuracy of our prediction and the confusion matrix.
    Uses the function in check_submission, and prints the accuracy,
    after that it finds the top ten tags the model has predicted wrong,
    and prints their confusion matrix.

    This function was used to check our accuracy and to print the confusion matrix for the report.
    '''
    accuracy,prob_sent,conf_pd = compare_files(true_file, pred_file)
    print(f"accuracy = {accuracy}")
    print("\n\n")

    # print conf_matrix
    conf_matrix = copy.deepcopy(conf_pd).to_numpy()
    diag_idx = np.diag_indices_from(conf_matrix)
    conf_matrix[diag_idx] = 0

    sum_tags = np.sum(conf_matrix, axis=1)
    max_ten_tags = np.argpartition(sum_tags, 10)[-10:]
    all_tags = conf_pd.index.to_list()

    print("confusion matrix of the ten tags the model failed on the most:")
    print(conf_pd.iloc[max_ten_tags, max_ten_tags])

def split_file(file_path):
    """
    Recieves a file path (mainly train2) and splits it for a train file and test file.
    This is used in order to create the files for cross validation of model 2,
    in order to check its accuracy, as there are no test files for it.

    This function is NOT used in the code of the assignment, 
    and is ONLY used to split to split the train2 file into two separate files.
    """
    data = []
    with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                data.append(line)

    train, test = train_test_split(data, test_size=0.75, random_state=3)

    f = open("mydata/train.wtag","w")
    for line in train:
        f.write(line + "\n")
    f.close()

    f = open("mydata/test.wtag","w")
    for line in test:
        f.write(line + "\n")
    f.close()


if __name__ == '__main__':
    true_file = 'mydata/test.wtag'
    pred_file = "mydata/opposite_yes_e_predictions_lam_0.5_thresh_(0.05, 'train2')_beam_2.wtag"
    main(true_file, pred_file)