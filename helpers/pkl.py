import pickle as pkl

with open('../../data/AIFB/graph_data.pkl', 'rb') as f:
    data = pkl.load(f)
print(data[2])