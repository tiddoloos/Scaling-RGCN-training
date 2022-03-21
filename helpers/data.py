from torch_geometric.datasets.entities import Entities

datasets = ['AIFB', 'MUTAG', 'AM']

def load_data():
    for dataset in datasets: 
        data = Entities(root='./data/', name=dataset)
        print(data)



# if __name__ == "__main__":
#     load_data()