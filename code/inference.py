import numpy as np
from preprocessing import *
from tqdm import tqdm
from scipy import sparse, special

def q_func(x, history, pre_trained_weights, feature2id):
    """ Calculates the q function of the Viterbi algorithm shown in class, for a given word, tag, and history.

    Args:
        x (str): current word
        history (tuple): (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
        pre_trained_weights: given weights of the model
        feature2id: the feature2id object

    Returns:
        np.array: the q value after softmax calculation.
    """
    tags = list(feature2id.feature_statistics.tags)
    input_to_softmax = list()
    for y_tag in tags:
        # Calculate the inner product of the weights and the features of the current word and tag, for all possible tags
        input_to_softmax.append(np.sum(np.array(pre_trained_weights[represent_input_with_features((x, y_tag) + history, feature2id.feature_to_idx)])))
    # Calculate q using the softmax function 
    softmax = special.softmax(np.array(input_to_softmax))
    return softmax


def word_tag_probability(feature2id):
    """ Calculates the probability of each tag given a word, using the probabilities dictionary in features statistics.

    Args:
        feature2id: the feature2id object

    Returns:
        dictionary: The probabilities of each tag given a word, in the form of a dictionary.
    """
    # Extract the probabilities dictionary from the feature statistics
    feature_prob_dict = feature2id.feature_statistics.feature_prob_dict['f113']
    prob_dict = {}
    # Iterate over the words and tags and calculate the probability of each tag given a word
    for c_word, value in feature_prob_dict.items():
        for c_tag in value.keys():
            if c_tag != "total":  # Skip the total key, which counts the total number of occurrences of the word
                prob_dict[(c_word,c_tag)] = value[c_tag] / value["total"]

    return prob_dict


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    tags = list(feature2id.feature_statistics.tags)
    b = 2  # Beam size
    beam_tags = range(len(tags))  # number of tags in this stage of the beam (initially all tags)
    prev_beam_tags = range(len(tags))  # number of tags in the previous stage of the beam (initially all tags)
    pi = np.ones(shape=(len(tags), len(tags)))  # Initialization of the pi matrix
    back_pointer = np.zeros(shape=(len(sentence), len(tags), len(tags))) # Initialization of the back pointer matrix.
                                                                         # In place i,j there will be the index of the tag, based on the tags set, 
                                                                         # that will represent the tag
    k = 0
    q = dict()
    w_t_prob = word_tag_probability(feature2id)  # A dictionary of the probabilities of each tag for every word. Key: (word, tag), Value: probability.
    for word in sentence:
        prev_pi = pi
        if k > 3:
            max_probs_indices = np.unravel_index(np.argsort(prev_pi.ravel())[-b:], prev_pi.shape)  # Get the indices of the b highest probabilities
            beam_tags = list(max_probs_indices[1])  # Get the tags of the b highest probabilities of the previous word
            prev_beam_tags = list(max_probs_indices[0]) # Get the tags of the b highest probabilities of the previous-previous word
            pi = np.zeros(shape=(len(tags), len(tags)))

        for i in beam_tags:
            for pp_tag in prev_beam_tags:  # Only iterate over the tags that are in the beam
                q[pp_tag] = q_func(word, (sentence[k-1] if k >= 1 else '*', tags[i], sentence[k-2] if k >= 2 else '*', tags[pp_tag], sentence[k+1] if k < len(sentence) - 1 else '~'), pre_trained_weights, feature2id)
            # We calculated the q function outside of the below loop to save time
            for j in range(len(tags)):
                pi[i][j], back_pointer[k][i][j] = max_on_t(prev_pi, i, j, word, tags, prev_beam_tags, pre_trained_weights, feature2id, q, w_t_prob)
                # Maximize the probability of the previous-previous tag. The back_pointer will hold the index of the tag that will represent the previous-previous tag.
        k += 1
    
    p_tag, c_tag = np.unravel_index(np.argmax(pi, axis=None), pi.shape)
    # Getting the most probable tag for the last word (while c_tag is *).

    our_tags = list()  # Our predicted tags
    our_tags.append(tags[p_tag])  # Append the most probable tag for the last word

    # Update our predicted tags using the back pointer matrix
    for k in range(len(sentence) - 3, 0, -1):
        pp_tag = int(back_pointer[k + 2][p_tag][c_tag])
        c_tag = p_tag
        p_tag = pp_tag
        our_tags.append(tags[pp_tag])
    
    our_tags.reverse()  # Reverse the list to get the tags in the correct order, from the first tag in the sentence to the last.
    return our_tags


def max_on_t(prev_pi, u, v, word, tags, prev_beam_tags, q_func, w_t_prob):
    """ Calculates the maximum probability of the previous-previous tag, and the tag that will represent it.

    Args:
        prev_pi (np.array): The pi matrix of the previous word
        u (int): The index of the previous tag
        v (int): The index of the current tag
        word (str): The current word
        tags (list): All possible tags
        prev_beam_tags (list): The tags that were in the beam in the previous stage
        q_func (np.array): The q function of the Viterbi algorithm
        w_t_prob (dictionary): The probabilities of each tag for every word

    Returns:
        float: The maximum probability of the previous-previous tag.
        int: The index of the tag that will represent the previous-previous tag.
    """
    max_p = 0
    max_probability_t = 0  # The index of the most probable pp-tag

    for t in prev_beam_tags:
        p = prev_pi[t][u] * q_func[t][v]  # Calculate the probability of the previous-previous tag, for all possible tags t, according to the Viterbi shown in class.
        
        # If the word (or the lower case version of it) and tag were seen in the training set,
        # use the probability from the w_t_prob dictionary.
        if (word, tags[v]) in w_t_prob.keys():
            p = prev_pi[t][u] * q_func[t][v] * w_t_prob[(word, tags[v])] 
        elif (word.lower(), tags[v]) in w_t_prob.keys():
            p = prev_pi[t][u] * q_func[t][v] * w_t_prob[(word.lower(), tags[v])] 

        # Update the maximum probability and its tag 
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



