#Importing all modules
from GoogleNews import GoogleNews
from newspaper import Article
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import math
googlenews = GoogleNews()

"""
We create a corpus consisting of text from googlenews documents.
The GoogleNews API is used to collect the articles. Each article is collected and added to the corpus.
"""

#topics chosen
topic_array = ['Health', 'Travel', 'India', 'Vaccines','Covid', 'Hospitals', 'Varients', 'Airport', 'Covaxin','Quarantine', 'Omicron']

corpus = []
document_count = 0
links=[]
print("The corpus is being created. Please refrain from running further blocks of code until the corpus is created.")

for topic in topic_array:
    googlenews.search(topic)
    for result in googlenews.results():
        article = Article(result['link'])
        try:
            article.download()
            article.parse()
            if len(article.text) > 0:
                corpus.append(article.text)
                links.append(result['link'])
                document_count += 1
            if document_count == 100:
                break
        except:
            continue
    googlenews.clear()

print("A corpus containing documents has been created.")

"""
Preprocessing the document corpus
"""
processed_corpus=[]
for document in corpus:

    #Pre-processing steps
    #Removal of digits from document
    document = ''.join(ch for ch in document if not ch.isdigit())

    #Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    document_without_punctuations = tokenizer.tokenize(document)

    #Normalization
    normalized_document = []
    for ele in document_without_punctuations:
        normalized_document.append(ele.lower())

    #Lemmatizing since its slightly better than Stemming
    lemmatizer = WordNetLemmatizer()
    lemmatized_document = []
    for ele in normalized_document:
        lemmatized_document.append(lemmatizer.lemmatize(ele))
        
    final_document = []
    stopwords = nltk.corpus.stopwords.words('english')
    for word in lemmatized_document:
        if word not in stopwords:
            final_document.append(word)

    processed_corpus.append(' '.join(final_document))

print("Corpus pre-processing done.")

"""
Total Vocabulary creation and creating a set of words in each document
"""

vocabulary=set()
doc_words=[set() for i in range(100)]
i=0
for document in processed_corpus:
    for word in document.split():
        vocabulary.add(word)
        doc_words[i].add(word)
    i+=1
print("Vocabulary created.")

"""
Calculating fij,Fi,N and link values
"""
corpus=processed_corpus
# calculating number of times ki occurs in the document dj
fij=defaultdict(lambda :0)
docid=0
for document in corpus:
    for word in document.split():
        fij[(word,docid)]+=1
    docid+=1

links_dict=defaultdict(lambda :0)
docid=0
for link in links:
    links_dict[docid]=link
    docid+=1
    
# frequency of occurence of term ki in the corpus
Fi=defaultdict(lambda : 0)
fijkeys=list(fij.keys())
for key in fijkeys:
    Fi[key[0]]+=fij[(key[0],key[1])]
    
N=len(corpus)
print("All values computed.")

'''
Query processing and standard values
'''
query="covid india"
query=query.split()
query_terms={}
termid=0
for term in query:
    query_terms[term]=termid
    termid+=1

print(query_terms)
#Standard values
K1=1
S1=(K1+1)
K2=0
K3=10
S3=(K3+1)

#Using BM25 model for ranking

avg_doclen=0
for document in corpus:
    avg_doclen+=len(document.split())
avg_doclen/=N

b=0.75
K1=1.2


bm25_val=[]
docid=0
for document in corpus:
    for term in query:
        final_val=0
        if term in document.split():
            relation_val=math.log2(abs((N-Fi[term]+0.5)/(Fi[term]+0.5)))
            Bij=(((K1+1)*fij[(term,docid)])/(K1*((1-b)+(b*(len(document.split())/avg_doclen)))+fij[(term,docid)]))
            val=Bij*relation_val
            final_val+=val
    bm25_val.append((final_val,docid+1))
    docid+=1

bm25_val.sort()
bm25_ranking=[]
bm25_links=[]

for j in range(N):
    bm25_ranking.append(bm25_val[j][1])
    bm25_links.append(links_dict[bm25_val[j][1]])
    
print('Ranking of documents and links done.')