from gensim.models import Word2Vec
# define training data
content = []
with open("data/sentences.train.small") as f:
    content = f.readlines()
sentences = []
for l in content:
    sentences.append(l.split(" "))
print (len(sentences))
print (sentences[0])
model = Word2Vec(sentences, min_count=1)
print (model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
#model.save('model.bin')
# load model
#new_model = Word2Vec.load('model.bin')
#print(new_model)
