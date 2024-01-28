import streamlit as st  
import emot
import string
import spacy
from googletrans import Translator
import re
import numpy as np
import pickle
from transformers import BertTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from textblob import TextBlob
from altair import Chart, X, Y, Color
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def clean_emoji(text):
    emot_obj = emot.core.emot()
    emoji = emot_obj.emoji(text)

    if emoji["flag"] == True:
        s = list(text)

        #unknow emoji
        if len(emoji['location']) != len(emoji['mean']):
            for n, (start, end) in enumerate(emoji['location'][::-1]):
                s[start:end] = [" " + " "]

        else:
            for n, (start, end) in enumerate(emoji['location'][::-1]):
                s[start:end] = [" " + emoji['mean'][-(n+1)] + " "]
        return "".join(s)

    else:
        return text

def clean_emoticons(text):
    emot_obj = emot.core.emot()
    emoji = emot_obj.emoticons(text)

    if emoji["flag"] == True:
        s = list(text)

        #unknow emoticons
        if len(emoji['location']) != len(emoji['mean']):
            for n, (start, end) in enumerate(emoji['location'][::-1]):
                s[start:end] = [" " + " "]

        else:
            for n, (start, end) in enumerate(emoji['location'][::-1]):
                s[start:end] = [" " + emoji['mean'][-(n+1)] + " "]
        return "".join(s)

    else:
        return text

def chat_words_conversion(text):
    chat_words = {
        "AFAIK": "as far as i know",
        "AFK": "away from keyboard",
        "ASAP": "as soon as possible",
        "ATK": "at the keyboard",
        "ATM": "at the moment",
        "A3": "anytime, anywhere, anyplace",
        "BAK": "back at keyboard",
        "BBL": "be back later",
        "BBS": "be back soon",
        "BFN": "bye for now",
        "B4N": "bye for now",
        "BRB": "be right back",
        "BRT": "be right there",
        "BTW": "by the way",
        "B4": "before",
        "CU": "see you",
        "CUL8R": "see you later",
        "CYA": "see you",
        "FAQ": "frequently asked questions",
        "FC": "fingers crossed",
        "FWIW": "for what it's worth",
        "FYI": "for your information",
        "GAL": "get a life",
        "GG": "good game",
        "GN": "good night",
        "GMTA": "great minds think alike",
        "GR8": "great!",
        "G9": "genius",
        "IC": "i see",
        "ICQ": "i seek you (also a chat program)",
        "ILU": "ilu: i love you",
        "IMHO": "in my honest opinion",
        "IMO": "in my opinion",
        "IOW": "in other words",
        "IRL": "in real life",
        "KISS": "keep it simple, stupid",
        "LDR": "long distance relationship",
        "LOL": "laughing out loud",
        "LTNS": "long time no see",
        "L8R": "later",
        "MTE": "my thoughts exactly",
        "M8": "mate",
        "NRN": "no reply necessary",
        "OIC": "oh i see",
        "PITA": "pain in the a..",
        "PRT": "party",
        "PRW": "parents are watching",
        "ROFL": "rolling on the floor laughing",
        "ROFLOL": "rolling on the floor laughing out loud",
        "SK8": "skate",
        "STATS": "your sex and age",
        "ASL": "age, sex, location",
        "THX": "thank you",
        "TTFN": "ta-ta for now!",
        "TTYL": "talk to you later",
        "U": "you",
        "U2": "you too",
        "U4E": "yours for ever",
        "WB": "welcome back",
        "WTF": "what the f...",
        "WTG": "way to go!",
        "WUF": "where are you from?",
        "W8": "wait",
    }

    new_text = []
    for w in text.split():
        if w.upper() in chat_words.keys():
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def Remove_url_id(text):
    pattern = r'https?://\S+|www\.\S+|@\S+'
    return re.sub(pattern, ' ', text)

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punct(text):
    """function adds a space for each punctuation"""
    punct = string.punctuation.replace("'", "")
    space = ' ' * len(punct)
    return text.translate(str.maketrans(punct, space))

def remove_ALL_punct(text):
    """function adds a space for each punctuation"""
    all_punct = string.punctuation
    all_space = ' ' * len(all_punct)
    return text.translate(str.maketrans(all_punct, all_space))

def lemmatize_words(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    lemmatized_sentence = ' '.join([token.lemma_ for token in doc])

    return lemmatized_sentence

def remove_stopwords(text):
    """function remove the stopwords"""
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    new_stopwords = ["would","shall","could","might"]
    stop_words.extend(new_stopwords)
    stop_words.remove("not")
    stop_words= list(set(stop_words))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def google_translate(text):
    translator = Translator()
    translator.raise_Exception = True
    return  translator.translate(text).text

def pre_pross(text):
    text = clean_emoji(text)
    text = clean_emoticons(text)
    text = chat_words_conversion(text)
    text = Remove_url_id(text)
    text = remove_html(text)
    text = text.lower()
    text = remove_punct(text)
    text = lemmatize_words(text)
    text = remove_stopwords(text)
    text = remove_ALL_punct(text)
    text = lemmatize_words(text)

    return text

def get_emoji(text):
    if text == "positive":
          return " 🟢👍🟢 "

    if text == "negative":
        return " ⛔👎⛔ "

    return " 😐😑 "

def get_style(output):
    if isinstance(output, str):
        # بررسی برای positive، negative و neutral
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'yellow'}
        return output

    else:  # catboost, XGboost
        if output.shape == (1,):  # XGboost
            output = output[0]

        if output == 0:
            return "negative"  # منفی
        elif output == 1:
            return "neutral"  # خنثی
        elif output == 2:
            return "positive"  # مثبت

    # Default values in case none of the conditions are satisfied
    return "unknown"

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def Roberta(text):
    '''
    Downloads the model from huggingface.
    '''
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # device = torch.cuda.current_device() if torch.cuda.is_available() else None
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    # return nlp
    
    # Preprocess text (username and link placeholders)

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    #model.save_pretrained(MODEL)
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    try:
      output = model(**encoded_input)
    except Exception as e:
      return {'negative': 0, 'neutral': 0, 'positive': 0}
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)

    ranking = ranking[::-1]
    d = {}
    for i in range(scores.shape[0]):

      l = config.id2label[ranking[i]]
      s = scores[ranking[i]]
      d[l] = np.round(float(s), 4)

    return d

def load_models_2():

    # MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # config = AutoConfig.from_pretrained(MODEL)
    # top_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    names = ['CART', 'MLP', 'SVC', 'Log_Reg', 'NB', 'catboost', "XGboost"]
    paths = [r'.\model_2\CART.pckl', 
             r'.\model_2\MLP.pckl', 
             r'.\model_2\svc.pckl', 
             r'.\model_2\Logistic_Regression.pckl', 
             r'.\model_2\nb.pckl', 
             r'.\model_2\catboost.pckl', 
             r'.\model_2\XGboost.pckl']
    
    models = {}
    for name, path in zip(names,paths):
        file = open(path, 'rb')
        models[name] = pickle.load(file)
        file.close()
    return models

def load_models_1():
    names = ['CART', 'MLP', 'SVC', 'Log_Reg', 'NB', 'catboost', "XGboost"]
    paths = [r'model\CART.pckl', 
             r'model\MLP.pckl', 
             r'model\svc.pckl', 
             r'model\Logistic_Regression.pckl', 
             r'model\nb.pckl', 
             r'model\catboost.pckl', 
             r'model\XGboost.pckl']
    
    models = {}
    for name, path in zip(names,paths):
        file = open(path, 'rb')
        models[name] = pickle.load(file)
        file.close()
    return models

def main():
    st.title("Sentiment Analysis")

    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key="form_1"):
            text = st.text_area("enter text")
            submit_button = st.form_submit_button(label='Analyze')
        
        tab_1 = st.container()
        tab_2 = st.container()
        # col_1, col_2 = st.columns(2)
        col_3, col_4 = st.columns(2)
        
        if submit_button and len(text.strip())>0:
            text = google_translate(text)
            text = str(TextBlob(text).correct())
            
            with tab_1:
                st.write("EN : "+text)
                text = pre_pross(text)
                st.write("After preprocess : "+text)
            
            with tab_2:
                st.info("roBERTa")
                x = Roberta(text)
                st.write(f'Positive🟢👍{x["positive"]}')
                st.write(f'Neutral 😐😑{x["neutral"]}')
                st.write(f'Negative⛔👎{x["negative"]}')
                data = {'Sentiment': ['positive', 'neutral', 'negative'],
                        'Probability': [x["positive"], x["neutral"], x["negative"]]}

                data = {'Sentiment': ['positive', 'neutral', 'negative'],
                        'Probability': [0.9661, 0.0275, 0.0064]}
                
                df = pd.DataFrame(data)

                chart = Chart(df).mark_bar().encode(
                    X('Sentiment:N', title='Sentiment'),
                    Y('Probability:Q', title='Probability'),
                    Color('Sentiment:N',
                        scale=Color(scale={'range': ['green', 'yellow', 'red']},
                            domain=['positive', 'neutral', 'negative'],
                            type='ordinal'),
                            title='Sentiment Color')
                    ).properties(
                        title='Sentiment Analysis',
                        width=400
                    )
                st.altair_chart(chart)#, use_container_width=True)
                
            
            with col_3:
                models_1 = load_models_1()
                st.info("Google 35k")

                for model_name, model in models_1.items():
                    p = model.predict([text])
                    target = get_style(p[0])
                    st.markdown(f'<span style="font-size:15px;">{model_name.capitalize()}: {get_emoji(target) + target}</span>', unsafe_allow_html=True)

            
            with col_4:
                st.info("batman 140k")
                models_2 = load_models_2()
                
                for model_name, model in models_2.items():
                    p = model.predict([text])
                    target = get_style(p[0])
                    st.markdown(f'<span style="font-size:15px;">{model_name.capitalize()}: {get_emoji(target) + target}</span>', unsafe_allow_html=True)
            
            text = ""
        
        elif submit_button and len(text.strip())==0:
            st.write("ERROR! enter a text")

    else:
        txt = """
            ### Sentiment Analysis Web App

            This web application performs sentiment analysis on the given text using various machine learning models. The application includes preprocessing steps such as cleaning emojis, emoticons, and chat words, as well as removing URLs, HTML tags, punctuation, and stop words. The main features of the app include:

            #### 1. Home Page:
            - **Text Input:** Users can input text in the provided text area.
            - **Analyze Button:** Clicking the "Analyze" button triggers the sentiment analysis process.

            #### 2. Sentiment Analysis:
            - **Preprocessing:** The entered text undergoes preprocessing, displaying both the original and preprocessed text.
            - **Robert Model:** Utilizes the [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model for sentiment analysis. Displays the probabilities for positive, neutral, and negative sentiments.
            - **Google 35k Models:** Includes various machine learning models trained on the Google 35k dataset. Displays the sentiment predictions and emojis for each model.
            - **Batman 140k Models:** Includes additional models trained on a larger dataset (Batman 140k). Displays the sentiment predictions and emojis for each model.

            #### Contributors:
            - Zohre Bagheri
            - Moein Zafari
            - Reza Naderi

            Feel free to input text and explore the sentiment predictions from different models."""      
        st.markdown(txt)

if __name__ == '__main__':
    main()
    
