import numpy as np
from preprocessing import read_test
from tqdm import tqdm


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """        
    feature_idx_dict = feature2id.feature_to_idx # * dictionary of all features from train with their index

    tags = list(feature2id.feature_statistics.tags) # TODO beam thingy

    q_func = lambda x, y: np.exp(np.dot(pre_trained_weights, create_feature_vector(x,y,feature_idx_dict))) \
                        / sum([np.exp(np.dot(pre_trained_weights, create_feature_vector(x,y_tag,feature_idx_dict)))\
                               for y_tag in tags]) # TODO add here the beam thingy

    prev_pi = np.ones(shape=(len(tags), len(tags))) # Initialization
    pi = np.zeros(shape=(len(tags), len(tags)))
    back_pointer = np.zeros(shape=(len(sentence), len(tags), len(tags))) # in place i,j there will be the index of the tag, based on the tags set, 
                                                                        # that will represent the tag
    k = 0
    for word in sentence:
        k += 1
        for i in range(len(tags)):
            u = tags[i]
            for j in range(len(tags)):
                v = tags[j]
                pi[i][j], back_pointer[k][i][j] = max_on_t(prev_pi, i, j, word, tags, k, q_func)
    
    p_tag, c_tag = np.argmax(pi)
    # previus: len - 2, current: len - 1

    our_tags = np.array(size=(len(sentence),))
    our_tags[-1] = tags[c_tag]
    our_tags[-2] = tags[p_tag]

    for k in range(len(sentence) - 3, -1, -1):
        pp_tag = back_pointer[k + 2][p_tag][c_tag]
        c_tag = p_tag
        p_tag = pp_tag
        our_tags[k] = tags[pp_tag]
    
    return our_tags

    # ! test cases, delete later
    # print("---------------------------------")
    # with open("file1.txt", "w") as file:
    #     file.write(str(feature_idx_dict))
    # with open("file2.txt", "a") as file:
    #     file.write(str(feature2id.feature_statistics.feature_rep_dict))
    # print("hello")
    # print(feature2id.feature_statistics.feature_rep_dict)
    # print("---------------------------")

    
    
def create_feature_vector(x,y, feature_idx_dict):
    features_on_x = [1 if (x,y) in feature_idx_dict[feature] else 0 for feature in feature_idx_dict.keys()]
    print("-------------------------------")
    print(features_on_x)
    print("-------------------------------")

    return np.array(features_on_x)

def max_on_t(prev_pi, u, v, word, tags, k, q_func):
    # TODO change word for sentence when finished with features
    max_p = 0
    max_t = 0

    for t in range(len(tags)):
        p = prev_pi[t][u] * q_func(word, v) # TODO change word according to features to resmbel the leacutre algorithem
        if max_p < p:
            max_p = p
            max_t = t
    return max_p, max_t
    


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
