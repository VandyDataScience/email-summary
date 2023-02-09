#!/usr/bin/env python
# coding: utf-8

# In[4]:


from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)


# In[5]:


def toneAnalyzer(obj):
    #after simple testing, writings with emotions across the board are more
    #suited towards nuetral, .4 might not be the correct -- am open to suggestions
    finalEmotion = 'neutral'
    print(obj, "\n")
    for item in obj:
        for item2 in item:
            #compare against highest
            if (item2['score'] > .4):
                 finalEmotion = item2['label']
    return(finalEmotion)


# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv('https://query.data.world/s/l3hzkdb27urlhvxxcdfkytmsjyrnai', on_bad_lines='skip')


# In[2]:


#need for preprocessing -not necessarily for tweets but for emails, i.e only the subject
#when you reply to emails the og email is there
classified = df.sample(1)['text'].to_string(buf = None, na_rep = "Neutral")

print(type(classified))
print(classified)


# In[6]:


testObj = classifier(classified)
print(toneAnalyzer(testObj))


# In[107]:


print(toneAnalyzer(testObj))


# In[ ]:


print(toneAnalyzer(testObj))

