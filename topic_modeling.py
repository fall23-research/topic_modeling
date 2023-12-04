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
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


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

query = f"""
                SELECT channel_name, freq_keywords
                FROM news_channel
                """
    
cursor_dest.execute(query)
keywords = cursor_dest.fetchall()

processed_data = []
for channel_name, freq_keywords in keywords:
    parsed_data = ast.literal_eval('[' + freq_keywords + ']')
    noun_list = [item[0] for item in parsed_data]
    processed_data.append(noun_list)

print(processed_data)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = corpora.Dictionary(processed_data)

# 빈도가 2 이상인 단어와 전체의 50% 이상 차지하는 단어 필터링
dictionary.filter_extremes(no_below=2, no_above=0.5)

# bag of words: 사전 속 단어가 문장에서 몇 번 출현하는지 빈도를 세서 벡터화
corpus = [dictionary.doc2bow(text) for text in processed_data]

num_topics = 5
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
pyLDAvis.save_html(lda_visualization, 'topic_modeling_visualization.html')


