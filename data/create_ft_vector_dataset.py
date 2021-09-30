import fasttext
import fasttext.util
import pandas as pd


fasttext.util.download_model('te', if_exists='ignore')
ft1 = fasttext.load_model('cc.te.300.bin')
ft2 = fasttext.load_model('indicnlp.ft.te.300.bin')

data = pd.read_json('annamayya_dataset_cleaned.json')
data['indic_ft_vectors'] = data.apply(lambda x: ft2.get_sentence_vector(x['Lyric']), axis=1)
data['ft_vectors'] = data.apply(lambda x: ft1.get_sentence_vector(x['Lyric']), axis=1)
data.to_json('annamayya_dataset_ft_vectors.json')
