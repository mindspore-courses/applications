with open("/home/ma-user/work/mindformers/ernie_model/ernie1.0/vocab.txt") as data:
    texts = data.readlines()

with open("/home/ma-user/work/mindformers/ernie_model/ernie1.0/vocab1.txt", "wr") as t:
    for text in texts:
        text = text.split('\t')[0]
        print(text)
        t.write(text)
        t.write('\n')