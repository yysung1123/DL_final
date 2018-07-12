from opencc import OpenCC
from pycorenlp import StanfordCoreNLP
import pickle

if __name__ == '__main__':
  converter = OpenCC('t2s')
  
  nlp = StanfordCoreNLP("http://localhost:9000")
  
  for i in range(1, 121):
    src_path = './source_data/' + str(i) + '.txt'
    src = open(src_path, 'r', encoding = 'utf-8')
	
    tagged_path = './tagged_data/' + str(i) + '.pkl'
    tagged = open(tagged_path, 'wb')
	
    for line in src:
      #converted = converter.convert(line)
      output = nlp.annotate(line, properties = {'annotators': 'pos', 'outputFormat': 'json'})
      print(output)
      pickle.dump(output, tagged)
      break
	  
    break
	
    src.close()
    tagged.close()
