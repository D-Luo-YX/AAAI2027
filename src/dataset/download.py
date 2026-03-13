from torch_geometric.datasets import Planetoid

dataset_names = ['Cora', 'CiteSeer', 'PubMed']

for name in dataset_names:
    print(f'Loading {name}...')
    dataset = Planetoid(root='data/Planetoid', name=name)
    data = dataset[0]
    print(f'{name}:')
    print(data)
    print('-' * 50)