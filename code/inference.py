import numpy as np
from preprocessing import *
from tqdm import tqdm
from scipy import sparse, special

def q_func(x, history, pre_trained_weights, feature2id):
    """
    Calculate the conditional probability of a tag given the sentence and the tag
    """
    tags = list(feature2id.feature_statistics.tags)
    # # Soft max of scipy - and if not, get the denominator out.
    # numerator = np.exp(np.sum(np.array([pre_trained_weights[i] for i in represent_input_with_features((x, y, history), feature2id.feature_to_idx)])))  # feature_to_idx: dictionary of all features from train with their index
    # denominator = np.sum(np.array([np.exp()]))  # TODO add here the beam thingy
    input_to_softmax = list()
    for y_tag in tags:
        input_to_softmax.append(np.sum(np.array(pre_trained_weights[represent_input_with_features((x, y_tag) + history, feature2id.feature_to_idx)])))
    # input_to_softmax = np.array([np.sum(np.array(pre_trained_weights[represent_input_with_features((x, y_tag, history), feature2id.feature_to_idx)])) for y_tag in tags])
    softmax = special.softmax(np.array(input_to_softmax))
    return softmax

def create_q_func_dict(c_word, history, pre_trained_weights, feature2id):
    """
    Create a dictionary of q functions for all possible tags in the sentence
    """
    q_func_dict = {}

    return q_func_dict

def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    tags = list(feature2id.feature_statistics.tags) # TODO beam thingy

    pi = np.ones(shape=(len(tags), len(tags)))  # Initialization
    back_pointer = np.zeros(shape=(len(sentence), len(tags), len(tags))) # in place i,j there will be the index of the tag, based on the tags set, 
                                                                         # that will represent the tag
    k = 0
    q = dict()
    for word in sentence:
        prev_pi = pi
        for i in range(len(tags)):
            for pp_tag in range(len(tags)):
                q[pp_tag] = q_func(word, (sentence[k-1] if k >= 1 else '*', tags[i], sentence[k-2] if k >= 2 else '*', tags[pp_tag], sentence[k+1] if k < len(sentence) - 1 else '~'), pre_trained_weights, feature2id)
            for j in range(len(tags)):
                pi[i][j], back_pointer[k][i][j] = max_on_t(prev_pi, i, j, word, tags, pre_trained_weights, feature2id, q)
        k += 1
    
    # p_tag, c_tag = np.argmax(pi)
    p_tag, c_tag = np.unravel_index(np.argmax(pi, axis=None), pi.shape)
    # previous: len - 2, current: len - 1

    our_tags = list()
    our_tags.append(tags[p_tag])

    for k in range(len(sentence) - 3, -1, -1):
        pp_tag = int(back_pointer[k + 2][p_tag][c_tag])
        c_tag = p_tag
        p_tag = pp_tag
        our_tags.append(tags[pp_tag])

    print(our_tags)
    our_tags.reverse()
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


def max_on_t(prev_pi, u, v, word, tags, pre_trained_weights, feature2id, q_func):
    # TODO change word for sentence when finished with features
    max_p = 0
    max_probability_t = 0  # The index of the most probable pp-tag
    # TODO history is currently only {}, change accordingly to the features

    for t in range(len(tags)):
        # p = prev_pi[t][u] * q_func[(word, v, dict(), pre_trained_weights, feature2id)] # TODO change word according to features to resmbel the leacutre algorithem
        p = prev_pi[t][u] * q_func[t][v]
        if max_p < p:
            max_p = p
            max_probability_t = t
    return max_p, max_probability_t
    


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



