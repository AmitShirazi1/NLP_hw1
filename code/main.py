import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from generate_comp_tagged import create_tagged_files

import time


def main():
    # MODEL 1
    train_path = "data/train1.wtag"
    test_path = "data/comp1.words"
    
    threshold = (1,"train1")
    lam = 0.5

    weights_path = 'weights.pkl'
    predictions_path = "comp_m1_314779166_325549681.wtag"

    statistics, feature2id = preprocess_train(train_path, threshold)

    t1 = time.time() # time before optimization
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
    print(f"time of optimization is: {time.time() - t1}")# time after

    create_tagged_files(test_path, weights_path, predictions_path)

    # MODEL 2
    train_path = "data/train2.wtag"
    test_path = "data/comp2.words"
    
    threshold = (0.05, "train2")
    lam = 0.5

    weights_path = 'weights.pkl'
    predictions_path = "comp_m2_314779166_325549681.wtag"

    statistics, feature2id = preprocess_train(train_path, threshold)

    t1 = time.time() # time before optimization
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
    print(f"time of optimization is: {time.time() - t1}")# time after

    create_tagged_files(test_path, weights_path, predictions_path)
    


if __name__ == '__main__':
    main()
