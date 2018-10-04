fix = pd.read_csv("/home/affine/Deep_Learning/FW__myntra_data/Spell_check/Corrected2.csv")
corrected = fix['Corrected'].tolist()
pp = fix['Pre_processed'].tolist()


def pre_processing(text):
    text=str(text).lower()
    text = re.sub(r'\b[am]+\b', '', text)
    tagged_sentence = nltk.tag.pos_tag(text.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    text = ' '.join(edited_sentence)
    #text = ' '.join([pprocess.data_loader.stemmer.stem(word) for word in text.split() if word not in pprocess.data_loader.cachedStopWords])
    return(text)


cc = []
pp2 = []
for i in corrected:
    cc.append(pre_processing(i))
for j in pp:
    pp2.append(pre_processing(j))

spell_checkTest = pd.DataFrame(list(zip(fix["Original"][:len(cc)], pp2, cc)))
spell_checkTest.columns = ["Original", "Pre_processed", "Corrected"]
# spell_checkTest = spell_checkTest[(spell_checkTest['Corrected'] != "")]
spell_checkTest.to_csv("/home/affine/Deep_Learning/FW__myntra_data/Spell_check/Corrected3.csv")
