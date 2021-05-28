[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![describtor - e.g. python version](https://img.shields.io/badge/Python%20Version->=3.6-blue)](www.desired_reference.com) ![](https://img.shields.io/badge/Software%20Mac->=10.14-pink)

# Letters from the Trenches 

## Topic Modelling of letters between loved ones sent during WWI


<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/letter%20(1).png"/></div>

Topic modelling is a form of unsupervised machine learning which is especially good at finding meaning in large, unstructured, text documents. The method has gained a lot of interest from a wide array of fields due to its potential for finding semantic information and visualising this in an appealing and understandable way.

In this assignment I took 100 letters exchanged between Australian, British, Canadian, and French soldiers and their loved ones as they fought along the front lines during WWI. In these times, writing was often the only form of communication and was surprisingly efficient given the hardships, with some records reporting delivery times of just a few days. These letters therefore represent a raw glimpse into the thoughts, hopes, and fears of these times. They tell us the stories of normal people living through incredibly pressing times. Some of the letters were written by soldiers in the moments before they headed over the top while others were the last records ever sent home.The collection includes 6 letters of British soldiers which passed uncensored and 20 letters of French soldiers which have been translated into English.

## Table of Contents 

- [Assignment Description](#Assignment)
- [Scripts and Data](#Scripts)
- [Methods](#Methods)
- [Operating the Scripts](#Operating)
- [Discussion of results](#Discussion)

## Assignment Description

**Applying unsupervised machine learning to text data** 

This assignment involved training an LDA model on data selected by oneself. The purpose was to extract structured information which could provide insight into the data. The approach would centre around topic modelling and could look at things such as whether authors would cluster together or how concepts would change over time. 

The assignment asked us to do the following: 

“Train an LDA model on your data to extract structured information that can provide insight into your data. You should formulate a short research statement explaining why you have chosen this dataset and what you hope to investigate. This only needs to be a paragraph or two long and should be included as a README along with the code.” 

**Purpose of the assignment:** 

This assignment is designed to test that you have an understanding of:
1.	How to formulate research projects with computational elements 
2.	How to perform unsupervised learning on text data 
3.	How to present results in an accessible manner 

**Personal focus of the assignment:** 

I took this assignment as an opportunity to work with a messy dataset that did not come with pre-cleaning, such as the Kaggle datasets. I did so to practice handling text data which is more similar to that found in the world of research, where regular expressions and extra processing steps must be applied to get the words into a workable format. I also focused on various visualisation techniques which can be found in the notebook “WW1_Topic_Modelling_Explained.ipynb”. To ensure the script was still legible, I did not include all of these visualisations in the final script but they can be viewed in the Output folder of the repository. 


## Scripts and Data 

**Scripts**

The assignment consists of one script and one notebook: 

| Script | Description|
|--------|:-----------|
WW1_Topic_Modelling_Explained.ipynb | Notebook containing full commentary on the steps taken through the topic modelling   
letters_topic_modelling.py   | A command-line script which can be run without any pre-processing

It is highly recommended that the reader looks through the notebook before running the script to get an idea of the pre-processing steps which have been taken and the the decisions along the way, such as what word types to include and what words to exclude from the topic modelling. 

**Data** 

The dataset used can be found in the 'data' folder by the name of '100_letters.csv'. Details of how this was collected can be found in the methods below. 

## Methods 

### Data collection 
The 100-letter data collection began with the following Kaggle [repository]( https://www.kaggle.com/anthaus/world-war-i-letters?select=letters.json). Here, a collection of 60 letters have been gathered into a json file. 20 of the letters are from French soldiers while the other 40 letters have been collected from the national archives in Britain. The first step of processing was to convert the file from json into a more readable csv format. From here, attempts were made to use pythons ‘googletrans’ package to translate the French letters into English. This was unfortunately not possible due to recent updates to Google’s API, with most online sources reporting they had been unable to get around the most recent update. Therefore, the letters were translated using Chrome’s Google translate and pasted back into the csv file with the other 40 letters.

The next step was to collect other online transcripts of letters exchanged during the war. Inclusion criteria ensured only letters exchanged between soldiers fighting along the front lines in France and their loved ones were included to stop the context being spread too far apart. This meant a number of letters from Australian soldiers posted to Gallipoli had to be excluded. The 40 additional letters were collected from news articles released around the 2018 centenary, more recent additions from the British national archive, the Australian government’s collection of transcribed documents, and a number of teacher’s educational sites. More documents would have been collected had they been available. 


### Data processing 

The data was loaded, transformed into a string variable, and cleaned using a number of regex steps. Due to a number of irregularities existing within the text such as “\\n\\” , some manual replacements were made to prepare the data for processing. The string variable was then passed through the SpaCy ‘nlp’ pipeline which tagged the words and organised the text into sentences. Sentences were gathered into chunks of 5, and passed through gensim’s bigram and trigram model functions, which parsed together phrases of 2 and 3 words commonly found together. The stop words were then removed using NLTK’s stop word library, the bi-gram and tri-grams were formed and spaCy lemmatization was used to keep only nouns, verbs, adjectives and adverbs. A gensim dictionary was then created to convert the words into an integer value and a corpus of all the words was created (this corpus is sometimes referred to as a ‘bag of words’ whereby words are taken and jumbled up to get rid of sentence structure and grammatical influence). 

### Building and Evaluating the model 

The model was built using gensim’s LdaMulticore function to have 5 defined topics, passing in chunk sizes of 5, and with a minimum probability threshold of 0. The perplexity and coherence scores were considered and multiple models were run to find the optimal combination of word types. These can be viewed in the “WW1_Topic_Modelling_Explained.ipynb” notebook and the final decision was taken to include only “nouns” and “adjectives” in the model. The nouns were considered necessary as they told the story of what people actually mentioned in terms of food substances, tools, gifts and such. Adjectives were included as they held much semantic information and were able to portray whether the letters were speaking with positivity and hope or fear and sadness. The verbs were considered also interesting for telling us what the people were doing, but they proved detrimental to the model coherence and so they were left out of the topic modelling.

Hyperparameter tuning was then used to determine the optimal number of topics to be included in the model, which was confirmed to be 5. The final steps involved visualising the data using a number of plotting methods such as word clouds, word count distributions, and pyLDAvis’s html visualisation function. Data frames were also created to present the number of topics, the keywords held within each, and examples of sentences from which they were formed. These can all be viewed in the notebook and also in the output folder.

## Operating the Scripts 

There are 4 steps to get the script up and running: 

***1. Clone the repository***

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository
git clone https://github.com/Orlz/Letters_From_Trenches.git
```

***2. Navigate into the correct repository***

From the terminal you can navigate into the repository with the following line: 


```bash
$ Letters_From_Trenches
```

***3. Create the virtual environment***

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal, navigate to the directory, and type the following code: 

```bash
bash create_virtual_environment.sh
```

And then activate the environment by typing:

```bash
$ source language_analytics03/bin/activate
```

***4. Run the Script***

There is just one script to run, which contains no parameters to define. This can be run from the terminal with the following code: 


```bash
$ python3 src/letters_topic_modelling.py
```

## Discussion of Results 

Five topics were created from our war letters, which were made up of both nouns and adjectives. The 10 keywords from each topic are visualised below in our word clouds:

<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/topic_wordclouds.jpeg"/></div>

So, what can we tell about these topics? The first observation is that there seems to be some level of sentiment which is playing a key role in the delineations. Topics 0 and 2 portray a feeling of positivity, love, and fellowship. Though mixed in with a few negative words such as “war” (topic 0), we get a raw glimpse of respite and consideration of time, with the words such as “day”, “month” and “year” appearing in topic 0, while “day” reappears in topic 2 alongside “time”, “rest”, “long” and “old”. These topics also give the impression of endearment, with their mentions of “fellow” and “love”. On the other hand, we are reminded that we are considering letters from a time of war in topic 1, where the inclusion of nouns can really be seen. The soldiers are talking of “guns”, “shells”, “trenches” and “lines”, which are likely from their descriptions of their current surroundings. Topic 4 is similar with giving the impression of a military context, but perhaps not so emotionally charged with more neutral words such as “field”, “country”, and “morning”.  It seems to represent the stories the soldiers would be writing home about, and their current movements. It is likely including verbs in the modelling would have presented in this topic 4 with words such as “go” and “move”. Topic 3 is a little harder to interpret, with more functional words such as “office” and “time”. Here we also see a glimpse of the communicative process, with the words “letter”, “last” and “present” suggesting they are acknowledging the communication from the sender.

Keywords alone are interesting, but they can’t tell us much about how the topics relate to one another. This is where the html visualisation can help to give insight into how well the topics are representing the data and how close in relation they are to one another. In topic modelling, you are trying to capture a topic in its own unique space, away from the others which indicates that this indeed where the words are uniquely clustering together. Too much overlap can say that too many topics are being created and the model is likely over fitting. Let’s have a look at how our topics present on a two dimensional space: 

 <img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/Screen%20Shot%202021-05-27%20at%2011.09.42%20am.png" alt="alt text" width="900" height="600"> 

Our topics seem to be distributed well, with no overlaps and bubbles forming across the dimensional scale, suggesting they are representing the data quite well. The bubbles are small, which is unsurprising considering the small dataset we’re using. It would be interesting to see how a bigger dataset would fare on this front. We can also use this visualisation to see how the various words are represented in each topic, and how this compares to the words frequency in the overall dataset. Of course, we see common words such as “time”, “day” and “letter” appearing in a lot of the topics, but the interesting places to look are the places where words are represented in a high proportion in this topic but not others. They are the words where we can start to extract meaning and insights from. 

_The reader is invited to look through the other visualisations which can be found in the Output folder_

In conclusion, we can see that topic modelling is certainly able to provide some structure to our letters from the war. We have been able to construct 5 topics which each give a little glimpse into what we would expect soldiers and their loved ones to have been talking about. It is interesting to consider that there are few mentions of “death” or “kill”, which would be expected to be a common topic of the time. Perhaps the soldiers wanted to avoid this topic, or spent their time writing home on more hopeful topics. Another interesting line of investigation could compare messages such as these to the written communication between soldiers in modern times and their loved ones. This could give new insights into the changing language of generations. All in all, the author hopes more documentation arises in text form that we can study from these times so that future projects can build the bigger datasets needed. 

***Teaching credit*** 👏

Many thanks to Ross Deans Kristiansen-McLachlen for providing an interesting and supportive venture into the world of Language Analytics! 

<div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
