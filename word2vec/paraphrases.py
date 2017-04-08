import re
import codecs

paraphrase_dir = '../resources/ppdb/m'
paraphrase_sources = {'lexical': 'ppdb-1.0-m-lexical',
'one_to_many': 'ppdb-1.0-m-o2m'}

class Paraphrases:
    def add(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
    
    def __init__(self, source):
        data_file = paraphrase_dir + '/' + paraphrase_sources[source]
        self.data = {}
        in_f = codecs.open(data_file, encoding='utf-8')
        for line in in_f:
            row = re.split(' \|\|\| ', line)
            key = row[1]
            value = row[2]
            #print(key, value)
            self.add(key, value)
        in_f.close()
                
    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return []
