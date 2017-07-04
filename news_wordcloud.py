# wordcloud subplots for all 4 news categories

# necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image

news = pd.read_csv('data/uci-news-aggregator.csv') # import dataset
news['TITLE'] = news['TITLE'].str.replace('[^\w\s]','') # unpunctuate

# create dataframe for each category
b_news = news.loc[news['CATEGORY'] == 'b'] # business
t_news = news.loc[news['CATEGORY'] == 't'] # science and technology
e_news = news.loc[news['CATEGORY'] == 'e'] # entertainment 
m_news = news.loc[news['CATEGORY'] == 'm'] # health

# convert news titles to usable strings for the word clouds
b_title = b_news['TITLE'].to_string()
t_title = t_news['TITLE'].to_string()
e_title = e_news['TITLE'].to_string()
m_title = m_news['TITLE'].to_string()

# import images and make them usable by word cloud
b_image = np.array(Image.open('images/business.jpg'))
t_image = np.array(Image.open('images/scitech.jpg'))
e_image = np.array(Image.open('images/entertainment.jpg'))
m_image = np.array(Image.open('images/health.jpg'))


fig = plt.figure(figsize=(15,12))

# setting stop-words, so words like "the" and "it" are ignored
stopwords = set(STOPWORDS)

# business news cloud
ax1 = fig.add_subplot(221)
b_wordcloud = WordCloud(background_color='white', mask=b_image, collocations=False, stopwords=stopwords).generate(b_title)
ax1.imshow(b_wordcloud, interpolation='bilinear')
ax1.set_title('business news', size=20)
ax1.axis('off')

# science and technology news cloud
ax2 = fig.add_subplot(222)
t_wordcloud = WordCloud(background_color='white', mask=t_image, collocations=False, stopwords=stopwords).generate(t_title)
ax2.imshow(t_wordcloud, interpolation='bilinear')
ax2.set_title('science & technology news', size=20)
ax2.axis('off')

# entertainment news cloud
ax3 = fig.add_subplot(223)
e_wordcloud = WordCloud(background_color='white', mask=e_image, collocations=False, stopwords=stopwords).generate(e_title)
ax3.imshow(e_wordcloud, interpolation='bilinear')
ax3.set_title('entertainment news', size=20)
ax3.axis('off')

# health news cloud
ax4 = fig.add_subplot(224)
m_wordcloud = WordCloud(background_color='white', mask=m_image, collocations=False, stopwords=stopwords).generate(m_title)
ax4.imshow(m_wordcloud, interpolation='bilinear')
ax4.set_title('health news', size=20)
ax4.axis('off')

plt.show()
