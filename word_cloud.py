from textblob import TextBlob 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from matplotlib import pyplot as plt 
import numpy as np 
from wordcloud import WordCloud 
from PIL import Image

lm = WordNetLemmatizer()


text  = """Technology ("science of craft") is the collection of techniques, skills, methods and processes used in the production of goods or services or in the accomplishment of objectives, such as scientific investigation. Technology can be the knowledge of techniques, processes, and the like, or it can be embedded in machines which can be operated without detailed knowledge of their workings.

The human species' use of technology began with the conversion of natural resources into simple tools. The prehistoric discovery of how to control fire and the later Neolithic Revolution increased the available sources of food and the invention of the wheel helped humans to travel in and control their environment. Developments in historic times, including the printing press, the telephone, and the Internet, have lessened physical barriers to communication and allowed humans to interact freely on a global scale. The steady progress of military technology has brought weapons of ever-increasing destructive power, from clubs to nuclear weapons.

Technology has many effects. It has helped develop more advanced economies (including today's global economy) and has allowed the rise of a leisure class. Many technological processes produce unwanted by-products known as pollution and deplete natural resources to the detriment of Earth's environment. Innovations have always influenced the values of a society and raised new questions of the ethics of technology. Examples include the rise of the notion of efficiency in terms of human productivity, and the challenges of bioethics.

Philosophical debates have arisen over the use of technology, with disagreements over whether technology improves the human condition or worsens it. Neo-Luddism, anarcho-primitivism, and similar reactionary movements criticise the pervasiveness of technology, arguing that it harms the environment and alienates people; proponents of ideologies such as transhumanism and techno-progressivism view continued technological progress as beneficial to society and the human condition.

Until recently, it was believed that the development of technology was restricted only to human beings, but 21st century scientific studies indicate that other primates and certain dolphin communities have developed simple tools and passed their knowledge to other generations."""


def word_graph(text):
	word_list = []

	tag_words = nltk.pos_tag(word_tokenize(text))
	#print tag_words
	#OUTPUT :
	#[('Next', 'JJ'), ('Steps', 'NNS'), ('that', 'IN'), ('we', 'PRP'), ('agreed', 'VBD'), ('up', 'RP'), ('on', 'IN'), ('1', 'CD'), ('.', '.'), ('Prateek', 'VB'), ('G', 'NNP'), ('to', 'TO'), ('create', 'VB'), ('a', 'DT'), ('spread', 'JJ'), ('sheet', 'NN'), ('with', 'IN'), ('all', 'DT'), ('available', 'JJ'), ('mails', 'NNS'), ('.', '.'), ('2', 'CD'), ('.', '.'), ('Matt', 'NNP'), (',', ','), ('Hessam', 'NNP'), (',', ','), ('Eswar', 'NNP'), ('will', 'MD'), ('call', 'VB'), ('out', 'RP'), ('which', 'WDT'), ('are', 'VBP'), ('positive', 'JJ'), ('and', 'CC'), ('which', 'WDT'), ('are', 'VBP'), ('negative', 'JJ'), ('.', '.'), ('(', '('), ('Kind', 'NNP'), ('of', 'IN'), ('crowd', 'NN'), ('sourcing', 'VBG'), (')', ')'), ('3', 'CD'), ('.', '.'), ('Prateek', 'VB'), ('G', 'NNP'), ('to', 'TO'), ('create', 'VB'), ('a', 'DT'), ('histogram', 'NN'), ('of', 'IN'), ('words', 'NNS'), ('per', 'IN'), ('mail', 'NN'), ('.', '.'), ('4', 'CD'), ('.', '.'), ('Compare', 'VB'), ('the', 'DT'), ('word', 'NN'), ('clouds', 'NN'), ('of', 'IN'), ('positive', 'JJ'), ('and', 'CC'), ('negative', 'JJ'), ('biased', 'VBN'), ('mails', 'NNS'), ('and', 'CC'), ('get', 'VB'), ('an', 'DT'), ('idea', 'NN'), ('of', 'IN'), ('what', 'WP'), ('is', 'VBZ'), ('defining', 'VBG'), ('this', 'DT'), ('.', '.'), ('We', 'PRP'), ('will', 'MD'), ('review', 'VB'), ('the', 'DT'), ('results', 'NNS'), ('after', 'IN'), ('this', 'DT'), ('step', 'NN'), ('and', 'CC'), ('decide', 'VB'), ('the', 'DT'), ('next', 'JJ'), ('steps', 'NNS'), ('.', '.')]

	for i in range(len(tag_words)):
		if(tag_words[i][1]=='JJ' ):#or tag_words[i][1]=="VB" or tag_words[i][1]=="RB" or tag_words[i][1]=="ADJ" or tag_words[i][1]=="ADV"):
			word_list.append(lm.lemmatize(tag_words[i][0]))

	#print word_list
	#OUTPUT :
	#['Next', 'spread', 'available', 'positive', 'negative', 'positive', 'negative', 'next']

	freq_words = nltk.FreqDist(word.lower() for word in word_list)
	#freq_words = FreqDist({'positive': 2, 'negative': 2, 'next': 2, 'available': 1, 'spread': 1})

	mask = np.array(Image.open("/home/aviso/Desktop/arrow.jpg"))
	wordcloud = WordCloud(background_color = 'white', mask = mask).generate(text)
	freq_words_polarity = [(word, TextBlob(word).sentiment[0]) for word in freq_words]
	#freq_words_polarity = [('available', 0.4), ('positive', 0.22727272727272727), ('spread', 0.0), ('negative', -0.3), ('next', 0.0)]


	words = []
	counts = []
	for word, count in freq_words.items():
		if(TextBlob(word).sentiment[0]>=0):
			count = count
		else :
			count = -count
		words.append(word) 
		counts.append(count)


	# Plot histogram using matplotlib bar().
	indexes = np.arange(len(words))
	width = 0.7
	plt.title("Word Counts")
	plt.xlabel("Words")
	plt.ylabel("Counts")
	plt.bar(indexes, counts, width)
	plt.xticks(indexes, words, rotation='vertical')
	plt.show()

	
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()

word_graph(text = text)