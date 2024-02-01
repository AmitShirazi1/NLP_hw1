import numpy as np
from preprocessing import read_test
from tqdm import tqdm
from scipy import sparse


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    feature_idx_dict = feature2id.feature_to_idx # * dictionary of all features from train with their index

    tags = list(feature2id.feature_statistics.tags) # TODO beam thingy

    q_func = lambda x, y: np.exp(np.dot(pre_trained_weights, create_feature100_vector(x,y,feature_idx_dict))) \
                        / sum([np.exp(np.dot(pre_trained_weights, create_feature100_vector(x,y_tag,feature_idx_dict)))\
                               for y_tag in tags]) # TODO add here the beam thingy

    prev_pi = np.ones(shape=(len(tags), len(tags))) # Initialization
    pi = np.zeros(shape=(len(tags), len(tags)))
    back_pointer = np.zeros(shape=(len(sentence), len(tags), len(tags))) # in place i,j there will be the index of the tag, based on the tags set, 
                                                                         # that will represent the tag
    k = 0
    for word in sentence:
        k += 1
        for i in range(len(tags)):
            for j in range(len(tags)):
                pi[i][j], back_pointer[k][i][j] = max_on_t(prev_pi, i, j, word, tags, k, q_func)
    
    p_tag, c_tag = np.argmax(pi)
    # previous: len - 2, current: len - 1

    our_tags = np.array(size=(len(sentence),))
    our_tags[-1] = tags[c_tag]
    our_tags[-2] = tags[p_tag]

    for k in range(len(sentence) - 3, -1, -1):
        pp_tag = back_pointer[k + 2][p_tag][c_tag]
        c_tag = p_tag
        p_tag = pp_tag
        our_tags[k] = tags[pp_tag]
    
    print(our_tags)
    
    return our_tags


def create_feature100_vector(x, y, feature_idx_dict):
    feature_vector = []
    for feature in feature_idx_dict['f100']:
        if feature == (x,y):
            feature_vector.append(1)
        else:
            feature_vector.append(0)

    return sparse.csr_matrix(feature_vector)


def create_feature101_vector():
    pass


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
