from pickle import load
from math import ceil
from itertools import product

from numpy import tanh
from scipy.spatial.distance import cosine
from pandas import DataFrame

#########################################
# Load Pickle Files                     #
#########################################
with open('data/papers.pkl', 'rb') as f:
    t = load(f)

with open('data/abs_wv.pkl', 'rb') as f:
    v = load(f)

with open('data/w_sets.pkl', 'rb') as f:
    s = load(f)

with open('data/wv_mod.pkl', 'rb') as f:
    m = load(f)


#########################################
# Average Function                      #
#########################################
def avg(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


#########################################
# Paper Pair Generator                  #
#########################################
def cp():
    for q in range(10):
        q_idx = f'PQ00{q + 1:02}'

        for c in range(10):
            c_idx = f'PC{q + 1:02}{c + 1:02}'

            yield q_idx, c_idx


#########################################
# Similarity Functions                  #
#########################################

# wmsimilarity
def _wmsimilarity(t1, t2):
    dist = m.wv.wmdistance(t1, t2)
    return 1 - tanh(dist)


def wmsimilarity(p1, p2):
    return _wmsimilarity(t[p1], t[p2])


# jaccard
def _jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)


def jaccard(p1, p2):
    return _jaccard(s[p1], s[p2])


# wordbyword
def _wordbyword(s1, s2, cutoff):
    ws1 = [w for w in s1 if w in m.wv.vocab]
    ws2 = [w for w in s2 if w in m.wv.vocab]

    cut = ceil(len(ws2) * cutoff)
    sim = []

    for w1 in ws1:
        s = [float(m.wv.similarity(w1, w2)) for w2 in ws2]
        s.sort(reverse=True)
        sim += s[:cut]

    return avg(sim)


def wordbyword(p1, p2, cutoff):
    return _wordbyword(s[p1], s[p2], cutoff)


# vectorcosine
def _vectorcosine(v1, v2):
    return cosine(v1, v2)


def vectorcosine(p1, p2):
    return _vectorcosine(v[p1], v[p2])


# wnsim_path
with open('data/wns_p.pkl', 'rb') as f:
    wnp = load(f)


def _wnsim_path(s1, s2, cutoff):
    cut = ceil(len(s2) * cutoff)
    sim = []

    for w1 in s1:
        s = [max(wnp[w1, w2]) if len(wnp[w1, w2]) > 0 else 0 for w2 in s2]
        s.sort(reverse=True)
        sim += s[:cut]

    return avg(sim)


def wnsim_path(p1, p2, cutoff):
    return _wnsim_path(s[p1], s[p2], cutoff)


# wnsim_wup
with open('data/wns_w.pkl', 'rb') as f:
    wnw = load(f)


def _wnsim_wup(s1, s2, cutoff):
    cut = ceil(len(s2) * cutoff)
    sim = []

    for w1 in s1:
        s = [max(wnp[w1, w2]) if len(wnw[w1, w2]) > 0 else 0 for w2 in s2]
        s.sort(reverse=True)
        sim += s[:cut]

    return avg(sim)


def wnsim_wup(p1, p2, cutoff):
    return _wnsim_wup(s[p1], s[p2], cutoff)


#########################################
# Run                                   #
#########################################
def dist100():
    dist = [[p, c,
            wmsimilarity(p, c),
            jaccard(p, c),
            wordbyword(p, c, 0.3),
            vectorcosine(p, c),
            ] for p, c in cp()]

    df = DataFrame(dist)
    df.columns = [
            'query', 'cited',
            'wmsimilarity',
            'jaccard',
            'wordbyword30',
            'vectorcosine',
    ]
    df.to_csv('data/out/dist100.csv', index=False)


def wndist():
    dist = [[p, c,
            wnsim_path(p, c, 0.3),
            wnsim_wup(p, c, 0.3),
            ] for p, c in cp()]

    df = DataFrame(dist)
    df.columns = [
            'query', 'cited',
            'path30',
            'wup30',
    ]
    df.to_csv('data/out/wnsim.csv', index=False)


#########################################
# Main                                  #
#########################################
if __name__ == '__main__':
    wndist()
