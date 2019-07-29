from pickle import dump
from sqlite3 import connect

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec


#########################################
# Generate Abstract Dictionary          #
#########################################
_STOPWORDS = frozenset(stopwords.words('english'))
_tokenizer = RegexpTokenizer(r'\w+')


def _load_content(con, ref):
    # get cursor
    cur = con.cursor()

    # get s2id
    s2id = cur.execute('SELECT s2id FROM Reference WHERE reference = ?',
                       (ref, )).fetchone()[0]

    # get title
    title = cur.execute('SELECT title FROM Paper WHERE s2id = ?',
                        (s2id, )).fetchone()[0]

    # remove stop words
    words = _tokenizer.tokenize(title)
    title = ' '.join([w.lower() for w in words if w.lower() not in _STOPWORDS])

    # get sentences
    query = 'SELECT content FROM SentProc WHERE abs_id = ? ORDER BY sent_id'
    sents = ' '.join([row[0] for row in cur.execute(query, (s2id, ))])

    return title + '. ' + sents


def get_abs(dbpath):
    # Dictionary - key for index, value for title & content
    papers = dict()

    # Creat DB connection
    con = connect(dbpath)

    # Get query papers
    for i in range(10):
        q_idx = f'PQ00{i + 1:02}'
        papers[q_idx] = _load_content(con, q_idx)

    # Get citations
    for q in range(10):
        q_num = f'{q + 1:02}'
        for c in range(10):
            c_num = f'{c + 1:02}'
            c_idx = f'PC{q_num}{c_num}'
            papers[c_idx] = _load_content(con, c_idx)

    # Close DB connection
    con.close()

    # key for index, value for title & content
    return papers


#########################################
# Generate Word Set for Papers          #
#########################################
def get_set(papers):
    # Dictionary - key for index, value for word set
    w_sets = dict()

    # For each paper
    for i, p in papers.items():
        w_sets[i] = set(p.split())

    # key for index, value for word set
    return w_sets


#########################################
# Generate Word Vector SUM for Papers   #
#########################################
_model = Word2Vec.load('newmodel')


def get_wvs(papers):
    # Dictionary - key for index, value for word vector
    abs_wv = dict()

    # For each paper
    for i, p in papers.items():
        # Compute the average
        wvs = [_model[w] for w in p.split() if w in _model.wv.vocab]
        abs_wv[i] = sum(wvs) / len(wvs)

    # key for index, value for word vector
    return abs_wv


#########################################
# ETL Run                               #
#########################################
if __name__ == '__main__':
    # papers.pkl
    papers = get_abs('papers.db')
    with open('papers.pkl', 'wb') as f:
        dump(papers, f)

    # w_sets.pkl
    w_sets = get_set(papers)
    with open('w_sets.pkl', 'wb') as f:
        dump(w_sets, f)

    # abs_wv.pkl
    abs_wv = get_wvs(papers)
    with open('abs_wv.pkl', 'wb') as f:
        dump(abs_wv, f)

    # wv_mod.pkl
    with open('wv_mod.pkl', 'wb') as f:
        dump(_model, f)
