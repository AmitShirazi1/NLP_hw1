import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def main():
    runs = []
    runs.append((50, 1))
    for i in range(2, 10, 2):
        for j in range(5, 20, 5):
            runs.append((i,float(j)/10))
    for j in range(5, 20, 4):
        runs.append((1,float(j)/10))
    for j in range(5, 20, 5):
        runs.append((5,float(j)/10))
    
    print(runs)
    print(len(runs))
    for run in runs:
        threshold = 0.10
        lam = 1
        
        train_path = "data/train1.wtag"
        test_path = "data/test1.wtag"

        weights_path = 'weights.pkl'
        predictions_path = f'pred/predictions_lam_{lam}_thresh_{threshold}.wtag'

        statistics, feature2id = preprocess_train(train_path, threshold)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
        # TODO: remove the comment from the line above

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        for feature in feature2id.feature_to_idx['f100']:
            if feature == ('The','DT'):
                print("yes")
        pre_trained_weights = optimal_params[0]
        # print('big matrix:', feature2id.big_matrix)
        # print('small matrix:', feature2id.small_matrix)

        print(pre_trained_weights)
        tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
        break
    


if __name__ == '__main__':
    main()