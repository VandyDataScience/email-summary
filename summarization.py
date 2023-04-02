from transformers import pipeline

model_name = 'csebuetnlp/mT5_multilingual_XLSum'

summarizer = pipeline("summarization", model=model_name, use_fast=False)

def preprocess(email_text):
  #figure out a preprocessing procedure
  return email_text
    
def generate_summary(email_text):
  #generates summary with max length varying based on input email length

  email_body = preprocess(email_text)

  num_of_words = len(email_body.split())

  if num_of_words < 30:
      return email_body
  elif num_of_words < 300:
    result = summarizer(email_body, min_length = int(num_of_words *0.3), max_length= int(num_of_words*0.7))
  else:
    result = summarizer(email_body, min_length = int(num_of_words *0.3), max_length= int(num_of_words*0.5))

  return result[0]['summary_text']