from gensim.models.word2vec import Word2Vec

def test(model, word):
    return model.most_similar(word)

if __name__ == '__main__':
    model = Word2Vec.load('word2vec.model')
    with open('test_word.txt') as f:
        word_list = f.read().split()
    for word in word_list:
        print(test(model, word))
