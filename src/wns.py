from pickle import load
from pickle import dump
from itertools import product
from nltk.corpus import wordnet
from sim import cp

#########################################
# Load Pickle File                      #
#########################################
with open('data/w_sets.pkl', 'rb') as f:
    s = load(f)


#########################################
# Path Similarity                       #
#########################################
wns_p = dict()

def wnsim_path(w1, w2):
    global wns_p

    sim = wns_p.get((w1, w2), None)
    if sim is not None:
        return sim

    set_1 = wordnet.synsets(w1)
    set_2 = wordnet.synsets(w2)

    sim = [wordnet.path_similarity(s1, s2) for s1, s2 in product(set_1, set_2)]
    sim = [s for s in sim if s is not None]

    wns_p[w1, w2] = sim
    wns_p[w2, w1] = sim
    return sim    


#########################################
# WuP Similarity                        #
#########################################
wns_w = dict()


def wnsim_wup(w1, w2):
    global wns_w

    sim = wns_w.get((w1, w2), None)
    if sim is not None:
        return sim

    set_1 = wordnet.synsets(w1)
    set_2 = wordnet.synsets(w2)

    sim = [wordnet.wup_similarity(s1, s2) for s1, s2 in product(set_1, set_2)]
    sim = [s for s in sim if s is not None]

    wns_w[w1, w2] = sim
    wns_w[w2, w1] = sim
    return sim 


#########################################
# Main                                  #
#########################################
if __name__ == '__main__':
    for p, c in cp():
        s1, s2 = s[p], s[c]
        for w1, w2 in product(s1, s2):
            _ = wnsim_path(w1, w2)
            _ = wnsim_wup(w1, w2)
    
    with open('wns_p.pkl', 'wb') as f:
        dump(wns_p, f)

    with open('wns_w.pkl', 'wb') as f:
        dump(wns_w, f)
