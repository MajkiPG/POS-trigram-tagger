import nltk
import numpy as np
from nltk.corpus import brown
from nltk import FreqDist
from collections import defaultdict
import statistics

#K-fold validation K definition
K = 10
#Laplace smoothing lambda parameter
lambda_var = 0.05

#Transition and emission functions
def transition_prob(u, v, s):
    return (fdist_trigram_tag[(u, v, s)]+lambda_var)/(fdist_bigram_tag[(u, v)]+lambda_var*len(tags))

def emission_prob(obs, state):
    return (fdist_tagged_words[(obs, state)]+lambda_var)/(fdist_tags[state]+lambda_var*len(tags))

def get_transition(x, y, z):
    try:
        return transitions[(x, y, z)]
    except KeyError:
        return lambda_var/(lambda_var*len(tags))

def get_emission(x, y):
    try:
        return emissions[x][y]
    except KeyError:
        return lambda_var/(lambda_var*len(tags))

#Get tagged words into a list
#Normalization - removed combined tags, all words to lower case
tagged_words = []
for tagged_word in brown.tagged_words():
    tagged_words.append((tagged_word[0].lower(), str(tagged_word[1]).split('-', 1)[0]))

print("Loaded " + str(len(tagged_words)) + " tagged words")
print(' ')

#K-folds validation loop
k_fold_results = []
for k in range(K):
    print('K-folds validation iteration nr: ' + str(k+1))
    print(' ')
    tagged_words_train = []
    tagged_words_test = []

    #Split data into training/testing
    #Takes first 100k words from brown corpus
    for i in range(100000):
        if int(i/10000) == k:
            tagged_words_test.append(tagged_words[i])
        else:
            tagged_words_train.append(tagged_words[i])
    
    tags = list(set([x[1] for x in tagged_words_train]))
    print(str(len(tags)) + " tags") 

    words = list(set([x[0] for x in tagged_words_train]))
    print(str(len(words)) + " unique words")

    bigrams = []
    bigrams += [(x, y) for x, y in nltk.bigrams(tagged_words_train)]
    print(str(len(bigrams)) + " bigrams")

    bigrams_set = list(set([(x[1], y[1]) for x, y in bigrams]))
    print(str(len(bigrams_set)) + " unique bigrams")

    trigrams = []
    trigrams += [(x, y, z) for x, y, z in nltk.trigrams(tagged_words_train)]
    print(str(len(trigrams)) + " trigrams")

    trigrams_set = list(set([(x[1], y[1], z[1]) for x, y, z in trigrams]))
    print(str(len(trigrams_set)) + " unique trigrams")

    print(' ')
    print('Calculating frequencies')
    fdist_tags = FreqDist([x[1] for x in tagged_words_train])
    fdist_bigram_tag = FreqDist([(x[1], y[1]) for x, y in bigrams])
    fdist_trigram_tag = FreqDist([(x[1], y[1], z[1]) for x, y, z in trigrams])
    fdist_tagged_words = FreqDist(tagged_words_train)

    tagged_words_set = list(set(tagged_words_train))
    possible_tags_dict = defaultdict(list)

    for tagged_word in tagged_words_set:
        possible_tags_dict[tagged_word[0]].append(tagged_word[1])
    
    print('Calculating transitions')
    transitions = {x: transition_prob(x[0], x[1], x[2]) for x in trigrams_set}

    print('Calculating emissions')
    emissions = {x: {y: emission_prob(x, y) for y in tags} for x in words}

    all_words = [x[0] for x in tagged_words_train]
    all_tags = [x[1] for x in tagged_words_train]

    observation = [x[0] for x in tagged_words_test]
    observation.append('STOP')

    desired_result = [x[1] for x in tagged_words_test]

    #Viterbi algorithm
    print(' ')
    print('Running Viterbi algorithm')
    viterbi = [[1.0 if i == 0 else 0.0 for i in range(len(observation)+2)] for j in range(len(bigrams_set)+2)]
    backpointers = [[' ' for i in range(len(observation)+2)] for j in range(len(bigrams_set)+2)]
    for t, word in enumerate(observation):
        for s, tag in enumerate(possible_tags_dict[word]):
            for sp, tagp in enumerate(bigrams_set):
                    new_score = viterbi[s][t] * get_transition(tagp[0],tagp[1], tag) * get_emission(word, tag)
                    if viterbi[sp][t+1] == 0.0 or new_score>viterbi[sp][t+1]:
                        viterbi[sp][t+1] = new_score
                    backpointers[sp][t+1] = tag
    
    #Backpropagate results from backpointers array
    print('Obtaining results')
    results = []
    for i in range(len(backpointers[0])-1, 0, -1):
        results.append((backpointers[np.argmax([x[i] for x in viterbi])][i]))
    results.reverse()
    results.pop()
    results.pop()

    #Calculate tagging accuracy
    k_fold_results.append(((desired_result==results).sum()/len(results))*100)
    print("Tagging accuracy result: " + str(k_fold_results[-1]) + "%")
    print(' ')

#Mean accuracy of k-folds validation
print('Mean accuracy: ' + str(statistics.mean(k_fold_results)) + '%c')
print(' ')  