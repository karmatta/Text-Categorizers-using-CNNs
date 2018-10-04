import enchant.checker as ec
import pandas as pd
import numpy as np
from ast import literal_eval
import re
import requests               # http://github.com/kennethreitz/requests
import nltk
from nltk.corpus import stopwords
import difflib
from nltk.stem import PorterStemmer as ps

nltk.download('averaged_perceptron_tagger')
ngram = 5

corpora = dict(eng_us_2012=17, eng_us_2009=5, eng_gb_2012=18, eng_gb_2009=6,
               chi_sim_2012=23, chi_sim_2009=11, eng_2012=15, eng_2009=0,
               eng_fiction_2012=16, eng_fiction_2009=4, eng_1m_2009=1,
               fre_2012=19, fre_2009=7, ger_2012=20, ger_2009=8, heb_2012=24,
               heb_2009=9, spa_2012=21, spa_2009=10, rus_2012=25, rus_2009=12,
               ita_2012=22)

def getNgrams(query, corpus, startYear, endYear, smoothing, caseInsensitive):
    params = dict(content=query, year_start=startYear, year_end=endYear,
                  corpus=corpora[corpus], smoothing=smoothing,
                  case_insensitive=caseInsensitive)
    if params['case_insensitive'] is False:
        params.pop('case_insensitive')
    if '?' in params['content']:
        params['content'] = params['content'].replace('?', '*')
    if '@' in params['content']:
        params['content'] = params['content'].replace('@', '=>')
    req = requests.get('http://books.google.com/ngrams/graph', params=params)
    res = re.findall('var data = (.*?);\\n', req.text)
    if res:
        data = {qry['ngram']: qry['timeseries']
                for qry in literal_eval(res[0])}
        df = pd.DataFrame(data)
        df.insert(0, 'year', list(range(startYear, endYear + 1)))
    else:
        df = pd.DataFrame()
    return req.url, params['content'], df


def runQuery(argumentString):
    arguments = argumentString.split()
    query = ' '.join([arg for arg in arguments if not arg.startswith('-')])
    if '?' in query:
        query = query.replace('?', '*')
    if '@' in query:
        query = query.replace('@', '=>')
    params = [arg for arg in arguments if arg.startswith('-')]
    corpus, startYear, endYear, smoothing = 'eng_2012', 1800, 2000, 3
    printHelp, caseInsensitive, allData = False, False, False

    # parsing the query parameters
    for param in params:
        if '-corpus' in param:
            corpus = param.split('=')[1].strip()
        elif '-startYear' in param:
            startYear = int(param.split('=')[1])
        elif '-endYear' in param:
            endYear = int(param.split('=')[1])
        elif '-smoothing' in param:
            smoothing = int(param.split('=')[1])
        elif '-caseInsensitive' in param:
            caseInsensitive = True
        else:
            print(('Did not recognize the following argument: %s' % param))
    if printHelp:
        print('See README file.')
    else:
        url, urlquery, df = getNgrams(query, corpus, startYear, endYear,
                                      smoothing, caseInsensitive)
        result = df.iloc[-1:]
        result = result.iloc[:, 2:]
        return result


def makeGram(sentence, error):
    befor_keyowrd, keyword, after_keyword = sentence.partition(error)
    befor_keyowrd = befor_keyowrd.split()
    if len(befor_keyowrd) == 0:
        return error
    else:
        il = befor_keyowrd[::-1]
        l = il[:ngram]
        l = l[::-1]
        l.append(str(error))
        return l

def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def fixSpelling(sentence, chkr, SpellDict):

    # Fix shorthand
    sentence = str(sentence).lower()
    chkr.set_text(sentence)
    for i in chkr:
        word = i.word
        if SpellDict[0].str.contains(word).any():
            i.replace(SpellDict.iloc[np.where(SpellDict[0].str.contains(word))[0][0]][1])
            sentence = i.get_text()

    # Fix typos
    chkr.set_text(sentence)
    for i in chkr:
        error = i.word
        # Get n-gram as a list
        gramN = makeGram(sentence, error)

        # Correct for only 4 - 5 grams
        if len(gramN) >= 5:
            # Concatenate list to n-gram
            gram = ' '.join(gramN[:len(gramN)-1])
            query = gram + str(" ? --startYear=1999 --endYear=2000")
            res = runQuery(query)
            while res.empty:
                gramN = gramN[1:]
                gram = ' '.join(gramN[:len(gramN) - 1])
                query = gram + str(" ? --startYear=1999 --endYear=2000")
                res = runQuery(query)

            cols = res.columns.tolist()
            ngram_sugg = [sugg.replace(str(gram) + " ", '') for sugg in cols]

            if len(list(set(ngram_sugg) & set(i.suggest()))) == 0:
                maxVal = float(res.max(axis = 1))
                fix = res.columns[(res == maxVal).iloc[0]][0]
                ngram_predicted_word = fix.split()[-1]

                suggestion = np.array(i.suggest())
                vSimilar = np.vectorize(similar)
                similarity = vSimilar(error, suggestion)

                ec_predicted_word = i.suggest()[similarity.argmax(axis=0)]
                ngramSimilarity = similar(ngram_predicted_word, error)
                ecSimilarity = similar(ec_predicted_word, error)
                if ngramSimilarity > ecSimilarity:
                    i.replace(ngram_predicted_word)
                else:
                    i.replace(ec_predicted_word)

            else:
                suggestion = np.array(ngram_sugg)
                vSimilar = np.vectorize(similar)
                similarity = vSimilar(error, suggestion)
                predicted_word = ngram_sugg[similarity.argmax(axis=0)]
                i.replace(predicted_word)
            sentence = i.get_text()
        else:  # bypass the n-gram suggestion for n-grams less than 4
            suggestion = np.array(i.suggest())
            vSimilar = np.vectorize(similar)
            similarity = vSimilar(error, suggestion)
            i.replace(i.suggest()[similarity.argmax(axis=0)])
            sentence = i.get_text()
    return str(sentence).lower()

def stem(sentence):
    for w in sentence:
        sentence = ' '.join(ps.stem(w))
        return sentence

def removeStopWords(text):
    stops = set(stopwords.words('english'))
    t = []
    for w in text.split():
        if w.lower() not in stops:
            t.append(w)
    text = ' '.join(t)
    return text

def pre_processing(text):
    text=str(text).lower()
    text = removeStopWords(text)
    text = re.sub(r'\b[u]\b', 'you', text)
    text = re.sub(r'\b[r]\b', 'are', text)
    text = re.sub(r'\b[y]\b', 'why', text)
    text = re.sub(r'\b[d]\b', 'the', text)
    text = re.sub(r'\b[n]\b', 'and', text)
    text = re.sub(r'\b[m]\b', 'am', text)
    text = re.sub(r'\b[b]\b', 'be', text)
    text = re.sub(r'\b[im]\b', 'i am', text)
    text = re.sub(r'\b([a-z]+[0-9]+|[0-9]+[a-z]+)\w*\b', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r'[.]{2,}', ' . ', text)
    text = re.sub(r'[.]', ' . ', text)
    text = re.sub(r'[!]{2,}', ' ! ', text)
    text = re.sub(r'[!]', ' ! ', text)
    text = re.sub(r'[?]{2,}', ' ? ', text)
    text = re.sub(r'[?]', ' ? ', text)
    text = re.sub(r'[,\-()]', ' ', text)
    text = re.sub(' +', ' ', text)
    tagged_sentence = nltk.tag.pos_tag(text.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    text = ' '.join(edited_sentence)
    #text = ' '.join([pprocess.data_loader.stemmer.stem(word) for word in text.split() if word not in pprocess.data_loader.cachedStopWords])
    return(text)

def main():

    src = "/home/affine/Deep_Learning/FW__myntra_data/Myntra-data-Category-fullpath.csv"
    datax = pd.read_csv(src)
    datax = datax.dropna()
    datax = datax.reset_index()
    feedback_n = [pre_processing(x) for x in datax["Feedback"]]

    chkr = ec.SpellChecker("en_US")
    chkr.add("myntra")
    SpellDict = pd.read_csv("/home/affine/Deep_Learning/FW__myntra_data/Spell_check/Spell_Check.csv", header=None)
    fix = []
    fix = pd.read_csv("/home/affine/Deep_Learning/FW__myntra_data/Spell_check/Corrected2.csv")
    fix = fix.drop(fix.columns[0], axis=1).drop(fix.columns[1], axis=1)
    fix = fix['Corrected'].tolist()

    try:
        for i in range(len(fix), len(feedback_n)):
            print(i)
            fix.append(fixSpelling(feedback_n[i], chkr, SpellDict))
    except:
        for i in range(len(fix), len(feedback_n)):
            print(i)
            fix.append(fixSpelling(feedback_n[i], chkr, SpellDict))

    #fixed = [fixSpelling(x, chkr, SpellDict) for x in datax["Feedback"]]


    spell_checkTest = pd.DataFrame(list(zip(datax["Feedback"][:len(fix)], feedback_n[:len(fix)], fix)))
    spell_checkTest.columns = ["Original", "Pre_processed", "Corrected"]
    #spell_checkTest = spell_checkTest[(spell_checkTest['Corrected'] != "")]
    spell_checkTest.to_csv("/home/affine/Deep_Learning/FW__myntra_data/Spell_check/Corrected2.csv")

if  __name__ =='__main__':main()


#very convinent and fast return service .



