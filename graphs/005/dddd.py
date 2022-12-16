import pickle


with open('edges_update.pkl','rb') as fin:
    data = pickle.load(fin)
print(len(data))
k = data['edges_ent2ent']
print(len(k[0]))
print(k[1][:,1])
