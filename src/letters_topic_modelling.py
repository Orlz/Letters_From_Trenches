#!/usr/bin/env python

"""
==============================================
Assignment 5: Topic Modelling with WW1 Letters 
==============================================

This script performs topic modelling on a dataset of 100 letters between French, British, Australian, and Canadrian soldiers and their loved ones during the First World War. 20 of the letters were originally in French and have been converted into English with the use of Google Translate (and some highschool French lessons! ;)) I hope you enjoy searching through the topics which these men and women chose to write about during the First World War. 

The script will work through the following steps: 
1.  Load in and format data into the correct 'type'
2.  Clean the data from unwanted characters and numbers 
3.  Generate bi-gram (a,b) and tri-gram (x, (a,b)) models using gensim 
4.  Create a gensim dictionary and corpus
5.  Build and run the LDA model 
6.  Compute the Perplexity and coherence scores 
7.  Creates a dataframe of the topics and their most frequent words 
8.  Create plots of word distribution per topic
9.  Create word clouds of the most common words in each topic
10. Generate interactive visualisation of topic models

The script can be run from the command line by navigating to the correct directory and environment, then typing: 
    $ python3 src/letters_topic_modelling.py 

""" 

"""
=======================
Import the Dependencies
=======================
"""

#operating systems 
import os
import sys
sys.path.append(os.path.join(".."))
from pprint import pprint

# data handling 
import pandas as pd
import numpy as np 
import re

#stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#Add some custom stopwords into nltk list
custom_stopwords = ["Dear", "dear", "Yours sincerely", "Kind regards", "Kindest regards", "sincerely", "Regards"]
stop_words = stopwords.words('english')
stop_words.extend(custom_stopwords)

# nlp functionality 
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])


# visualisations 
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10

#LDA tools 
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import Letters_from_TrenchesOLD.utils.lda_utils as lda_utils


# warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


"""
==============
Main Function 
==============
"""

def main():
    
    #Tell the user you're about to start up the process
    print("\nHello, I'm setting up your letters from WW1 topic modelling...")
    
    """
    -------
    Step 1: Loading the data
    -------
    """
    # create the output directory
    if not os.path.exists("output"):
        os.mkdir("output")
        
    # mute the warning notifications
    warnings.filterwarnings("ignore")
    
    #Load in the data
    print("\nCurrently loading in the data and about to clean it up for you!") 
    data_path = os.path.join("data", "100_Letters.csv")
    
    #Convert the data into a string variable
    data = pd.read_csv(data_path)   #first load it as a dataframe
    text = data['text']             #separate off the column which contains the text 
    
    text_list =[]                   #Make empty list to append the data to 

    for line in text:               #Iterate over the text object and append each line of text into the empty list
        text_list.append(line)
    
    #Finally we make the list a string variable 
    #The join() function here needs to be reinforced by the map() function                                  
    string = (', '.join(map(str, text_list)))
    
    """
    -------
    Step 2: Cleaning the data
    -------
    """
    
    #We'll start with a regex expression first to catch most of the characters
    cleaned_text = re.sub(r'[^\w\s\n\\n]', '', string)
    
    #Then, we need to manually clean some unnusual noise which has slipped through
    cleaned_text = cleaned_text.replace("\\n\\n", " ")
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\'", " ")
    cleaned_text = cleaned_text.replace("\'t", " ")
    cleaned_text = cleaned_text.replace("\\n", " ")
    
    #Finally, we want to remove all the digits which are common to find within the letters 
    #These won't add anything to our analysis so we'll replace them with a space 
    pattern = r'[0-9]'
    cleaned_text = re.sub(pattern, '', cleaned_text)
    
    """
    -------
    Step 3: Generate bi-gram and tri-gram models with gensim 
    -------
    """
        
    print("\nData cleaning complete. I'm about to process the data and generate your bi and tri grams")
        
    #Cleaned_text is not a good variable name so we'll rename it 
    text = cleaned_text
    
    #The first step is to run out text through spacy to give it word labels
    #And to split into sentences 
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    
    #Our sentences are small in size so we'll lump them together into chunks of 5 sentences at a time
    #This is still small, but bigger wouldn't leave us with much data!
    chunks = []
    
    for i in range(0, len(sentences), 5):
        chunks.append(' '.join(sentences[i:i+5]))
    
    #Now we can call our gensim_processing function on our chunks of data to create our bigrams and trigrams
    #This returns our data as a list which can be used to make a dictionary and corpus 
    data_processed = gensim_processing(chunks)
    
    """
    -------
    Step 4: Create a gensim dictionary and corpus  
    -------
    """
    print("\nAll done, the data is ready. Now, I'll create the dictionary and corpus")
    
    #Call the create_dict_corpus function 
    dictionary, corpus = create_dict_corpus(data_processed)
    
    
    """
    -------
    Step 5: Build the LDA model   
    -------
    """
    print("\nThe dictionary and corpus have been created. I'll start building  your LDA model with 5 topics...") 
    
    #We'll build an LDA model with 5 topics (the optimal number according to our hypertuning) 
    lda_model = gensim.models.LdaMulticore(corpus=corpus,            #our corpus 
                                           id2word=dictionary,       #our dictionary 
                                           num_topics=5,             #our number of topics defined as 15
                                           random_state=42,          #the number of random states (helps with repdoducability)
                                           chunksize=5,              #chunck size to help model be more effifienct 
                                           passes=10,                #Number of times the model passes through the data 
                                           iterations=100,           #We want the model to iterate over the data 100 times
                                           per_word_topics=True,     #We want to assign each word topic
                                           minimum_probability=0.0) ##Topics with threshold lower than this will be filtered
    
    """
    -------
    Step 6: Compute the Perplexity and Coherence scores    
    -------
    """
    print("Calculating perplexity and coherence...") 
    
    """
    Perplexity =  A measure of how good the model is. The lower the number the better. 
    Coherence =   A measure of how semantically similar the top scoring words in a topic are.
    """
    
    # Compute Perplexity
    perplexity = lda_model.log_perplexity(corpus)
    
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model= lda_model, 
                                 texts= data_processed, 
                                 dictionary= dictionary, 
                                 coherence= 'c_v')
    coherence = coherence_model_lda.get_coherence()
    
    print (f"\n The perplexity is {perplexity} and the coherence is {coherence}.") 
    
    #We'll save these as a txt file 
    topic_words = lda_model.print_topics()
    with open('output/topic_info.txt', 'w+') as f:
        f.writelines(f"The perplexity score is: {perplexity}, The coherence score is: {coherence}\n\n")
        f.writelines("The 5 Topics are summarised as follows:\n\n")
        f.writelines(str(topic_words))
        
    print("\nThis information is saved in the output folder as 'topic_info.txt'.")
    
    
    """
    -------
    Step 7: Create a dataframe of the topics and their keywords 
    -------
    """
    print("Creating a txt file with the output topics. This will be found in output")
    
    #Here we look closer at the topics made and create a dataframe of these 
    #print the topics to the terminal 
    pprint(lda_model.print_topics())
    
    
    #Create a data_frame of the topic keywords
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                          corpus=corpus, 
                                                          texts=data_processed)
    
    #Define the dominent topics for later 
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # Ammend the table dimensions to make this into a presentable table 
    pd.options.display.max_colwidth = 100

    #sort the topics into their dominant topic (in order) 
    sent_topics_sorteddf = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                          grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                          axis=0)

    # Reset Index    
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)

    # Create the column headings for the dataframe
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    sent_topics_sorteddf.to_csv("output/topic_keywords.txt", index = False)
    
    
    """
    -------
    Step 8: Create plots of word distribution per topic 
    -------
    """   
    print("\nNow we'll start creating some plots...") 
    print("\nFirst I'll create frequency distribution of word counts in the sentence chunks") 
    
    #Here I develop upon some code created by www.machinelearningplus.com and create distributions of word frequency
    #It will create 5 plots (1 for each topic) 

    #define the colour scheme
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] #using the colours from tableau (data viz platform) 
    
    #Set up the axes and tell the plot there's going to be multiple plots in one 
    fig, axes = plt.subplots(5,1,figsize=(8,12), dpi=140, sharex=True, sharey=True)
    
    #Determine the asthetics of the plot 
    for i, ax in enumerate(axes.flatten()):    
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins = 30, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 60), xlabel='Word Count')
        ax.set_ylabel('Number of Chunks', color=cols[i])
        ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))
   
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0,60,9))
    fig.suptitle('Distribution of Word Counts by Topic', fontsize=22)
    
    #save the plot 
    fig.savefig("output/word_frequency_plot.png")
    
    
    """
    -------
    Step 9: Create word clouds of the most common words in each topic  
    -------
    """
    print("\nNext we'll create some wordclouds for each topic")                    
    
    #define the colour scheme                   
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    #determine the asthetics of the wordcloud                    
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    #set the number of topics to 5                   
    topics = lda_model.show_topics(num_topics = 5, formatted=False)

    fig, axes = plt.subplots(1, 5, figsize=(20,15), sharex=True, sharey=True)

    #generate the clouds
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    # save wordclouds 
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig("output/topic_wordclouds.jpg")
                       
    """
    -------
    Step 10: Generate interactive visualisation of topic models 
    -------
    """ 
    print("Finally, creating your interactive html") 
                       
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, f"output/Topic_Modelling_viz.html")                
                       
    
    #Tell the user your script is finished 
    print("That's you finished, enjoy the results!")
    
    
"""
-----------
Functions 
-----------
""" 

"""
Here we have 4 functions which are called above to keep the topic modelling steps more modular and easy to read. 
Each of the functions below has it's own description to describe the steps it's taking 
"""
    
def gensim_processing(data):
    """
    Here we use gensim to define bi-grams and tri-grams which enable us to create a create a dictonary and corpus
    We then process the data by calling the process_words function from our utils folder
    """
    #build the models first 
    bigram = gensim.models.Phrases(data, min_count=3, threshold=15) #We're lowering the threshold as we've not a lot of data 
    trigram = gensim.models.Phrases(bigram[data], threshold=15)  
    
    #Then fit them to the data 
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    #We further process the data using spacy and allow Nouns and Adjectives to pass (not verbs or adverbs!) 
    data_processed = lda_utils.process_words(data,nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN","ADJ"])

    #We now have a list of words which can be used to train the LDA model
    return data_processed


def create_dict_corpus(data_processed):
    """
    Here we create a dictonary and a corpus. 
    => The dictionary converts the words into an integer value
    => The corpus creates a 'bag of words' model for all the data (i.e. mixes it up and makes it unstructured) 
    
    """
    # Create Dictionary
    dictionary = corpora.Dictionary(data_processed)
    
    #We want to remove very common words so we'll filter those which appear in more than 80% of the letters
    dictionary.filter_extremes(no_above=0.80)  

    # Create Corpus: Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in data_processed]
    return dictionary, corpus 


if __name__=="__main__":
    #execute main function
    main()
