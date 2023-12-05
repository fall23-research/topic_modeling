# -*- coding: utf-8 -*-

# DB 연결
from konlpy.tag import Okt
okt = Okt()
from dotenv import load_dotenv
load_dotenv()
import os
import mysql.connector
import ast

# LDA 모델
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric
import logging

# 시각화
import pickle
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import pyLDAvis.gensim


DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

config = {
    'user': DB_USER,
    'password': DB_PASSWORD,
    'host': DB_HOST,
    'database': DB_NAME,
    'raise_on_warnings': True,
}

conn_dest = mysql.connector.connect(**config)
cursor_dest = conn_dest.cursor()

official_news_channel_names = ['SBS 뉴스', 'MBCNEWS', 'JTBC News', 'KBS News', '채널A 뉴스', 'MBN News', '뉴스TVCHOSUN']

official_news_channel_ids = ['UCkinYTS9IHqOEwR1Sze2JTw', 'UCF4Wxdo3inmxP-Y59wXDsFw', 'UCsU-I-vHLiaMfV_ceaYz5rQ', 'UCcQTRi69dsVYHN3exePtZ1A', 'UCfq4V1DAuaojnr2ryvWNysw', 'UCG9aFJTZ-lMCHAiO1KJsirg', 'UCWlV3Lz_55UaX4JsMj-z__Q']

query = f"""
            SELECT freq_keywords
            FROM news_video
            WHERE channel_id = "{official_news_channel_ids[6]}"
        """
    
cursor_dest.execute(query)
keywords = cursor_dest.fetchall()

processed_data = []
for keyword in keywords:
    freq_keywords = keyword[0]
    parsed_data = ast.literal_eval('[' + freq_keywords + ']')
    noun_list = [item[0] for item in parsed_data]
    processed_data.append(noun_list)

print(processed_data)

def replace_word(word):
    word = '확진자' if word == '진자' else word
    word = '거리두기' if word == '거리' else word
    word = '' if word == '두기' else word
    word = '지원금' if word == '원금' else word
    word = '' if word == '일단' else word
    word = '' if word == '지난' else word
    word = '' if word == '정말' else word
    word = '' if word == '진짜' else word
    word = '' if word == '그냥' else word
    word = '확진자' if word == '확정자' else word
    word = '' if word == '저희' else word
    word = '' if word == '여러분' else word
    word = '대한민국' if word == '대한' else word
    word = '' if word == '제발' else word
    word = '아스트라제네카' if word == '아스' else word
    return word

# 이중 리스트를 순회하면서 단어 변경
for i in range(len(processed_data)):
    for j in range(len(processed_data[i])):
        processed_data[i][j] = replace_word(processed_data[i][j])


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = corpora.Dictionary(processed_data)

# 빈도가 2 이상인 단어와 전체의 50% 이상 차지하는 단어 필터링
dictionary.filter_extremes(no_below=2, no_above=0.5)

# bag of words: 사전 속 단어가 문장에서 몇 번 출현하는지 빈도를 세서 벡터화
corpus = [dictionary.doc2bow(text) for text in processed_data]

num_topics = 3
chunksize = 2000
passes = 20
iterations = 400
eval_every = None

temp = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

lda_visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_visualization, 'TVCHOSUN_News_topic_modeling.html')