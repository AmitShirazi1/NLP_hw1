from check_submission import compare_files
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def main(true_file, pred_file):
    '''
    Prints the accuracy of our 
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