# -*- coding: utf-8 -*-

from konlpy.tag import Okt
okt = Okt()
from collections import Counter
import json
from dotenv import load_dotenv
load_dotenv()
import os
import mysql.connector
import ast

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
