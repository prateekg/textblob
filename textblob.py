from textblob import TextBlob
from textblob import Word

text = "" #get the text 

#1.create TextBlob object
blob = TextBlob(text)

#2. Part-of-Speech Tagging
blob.tags

#3. Noun-Phase extraction
blob.noun_phase()

#4. Sentiment Analysis
blob.sentiment #reutrn a tuple of (polarity, subjectivity) where polarity lies between [-1, 1] and subjectivity lies between [0, 1]

#5. Word and Sentence tokenize
blob.words #returns a iterable list of words 
blob.sentences #returns a iterable list of sentences

#6. Lemmatization
w = Word(text)
w.lemmatize() #we can pass in part of speech for particular lemmatization

#7. Wordnet
from textblob.wordnet import VERB, Synset
word = Word(text)
word.synsets
word.get_synsets(pos=VERB)

#8. Spelling correction
blob.correct() #it will return the correct word with the highest confidence
w = Word(text)
w.spellcheck() #it will return the list of tuple (correct_word, confidence)

#9. Word and NounPhase Frequencies 
blob.word_counts("____", case_sensitive = True) #bydefualt case_sensitive = False
blob.words.count("____")
blob.noun_phrases.count("____")

#10. Language Translation and Detection
blob.translate(to = "")
blob.detect_language()

#11. N-Grams
blob.ngrams(n=3) #returns a tuple of 3 successive words



