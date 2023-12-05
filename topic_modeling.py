# -*- coding: utf-8 -*-

# DB 연결
from konlpy.tag import Okt
okt = Okt()
from dotenv import load_dotenv
load_dotenv()
import os
import mysql.connector
import ast

# preprocessing
from konlpy.tag import Okt
from collections import Counter

# LDA 모델
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import logging

# 시각화
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

official_news_channel_names = ['SBS', 'MBC', 'JTBC', 'KBS', 'ChannelA', 'MBN', 'TVCHOSUN']
official_news_channel_ids = ['UCkinYTS9IHqOEwR1Sze2JTw', 'UCF4Wxdo3inmxP-Y59wXDsFw', 'UCsU-I-vHLiaMfV_ceaYz5rQ', 'UCcQTRi69dsVYHN3exePtZ1A', 'UCfq4V1DAuaojnr2ryvWNysw', 'UCG9aFJTZ-lMCHAiO1KJsirg', 'UCWlV3Lz_55UaX4JsMj-z__Q']

with open('stopwords.txt', 'r', encoding='utf-8') as f:
  stopwords = [line.strip() for line in f]

def replace_word(word):
        word = '확진자' if word == '진자' else word
        word = '거리두기' if word == '거리' else word
        word = '지원금' if word == '원금' else word
        word = '확진자' if word == '확정자' else word
        word = '대한민국' if word == '대한' else word
        word = '아스트라제네카' if word == '아스' else word
        word = '코로나' if word == '코로' else word
        word = '코로나' if word == '코론' else word
        word = '격리' if word == '경리' else word
        return word

counter = 0
for news_channel in official_news_channel_ids:
    query = f"""
                SELECT transcript
                FROM news_video
                WHERE channel_id = "{news_channel}"
                ORDER BY RAND()
                LIMIT 500;
            """
        
    cursor_dest.execute(query)
    transcripts = cursor_dest.fetchall()

    processed_data = []
    for transcript_tuple in transcripts:
        transcript = transcript_tuple[0]
        tokens = okt.pos(transcript, stem=True, norm=True)

        # Filter nouns and adjectives, excluding stopwords
        filtered_tokens = [word for word, pos in tokens if pos in ['Noun'] and len(word) > 1 and word not in stopwords]
        replaced_filtered_tokens = [replace_word(word) for word in filtered_tokens]
        word_freq = Counter(replaced_filtered_tokens)

        # Extracting words from the Counter and storing them in a list
        frequent_words_list = list(word_freq.keys())
        
        processed_data.append(frequent_words_list)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dictionary = corpora.Dictionary(processed_data)

    # 빈도가 2 이상인 단어와 전체의 50% 이상 차지하는 단어 필터링
    dictionary.filter_extremes(no_below=2, no_above=0.4)

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

    top_topics = model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)

    lda_visualization = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_visualization, f'{official_news_channel_names[counter]}_News_topic_modeling.html')
    counter += 1