import pickle
import torch
import numpy as np
#pred = open('pred_counts.pkl','rb')
#data = pickle.load(pred)
#for i in range(0,51):
#    print(data[i],'/',i)
#print(len(data[1]),len(data[1][1]))

num_img_pred = 51
num_img_ent = 151
rel_inds = np.zeros((51,2))
edges_img_pred2subj = np.zeros([num_img_pred, num_img_ent])   #给出场景图中subj/obj和predicate之间的边
#edges_img_pred2subj[torch.arange(num_img_pred), rel_inds[:, 0]] = 1
#print(edges_img_pred2subj.size)


t = torch.arange(1,6)
print(torch.topk(t,k=3,dim=0)[1])
#t.scatter_(dim=0 , torch.topk(t,k=3,dim=1)[1],0)
print(t)