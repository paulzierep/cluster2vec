import random
import csv

#################
#creata random clusters
#################

def get_item_list(lis, num_items, max_cluster_size, min_cluster_size):
	items = []
	for i in range(num_items):
		item = [lis[random.randint(0, len(lis) - 1)] for x in range(random.randint(min_cluster_size, max_cluster_size))]
		items.append(item)

	return(items)

pks = ['AT', 'KS', 'ACP']
nrps = ['A', 'E', 'C']
mix = ['AT', 'KS', 'ACP', 'A', 'E', 'C']

max_cluster_size = 40
min_cluster_size = 4

num_items = 2000

data_dict = {}

data_dict['pks'] = get_item_list(pks, num_items, max_cluster_size, min_cluster_size)
data_dict['nrps'] = get_item_list(nrps, num_items, max_cluster_size, min_cluster_size)
data_dict['mix'] = get_item_list(mix, num_items, max_cluster_size, min_cluster_size)

main_list = []

index = 0
for cla, text in data_dict.items():
	for sent in text:
		sub_list = [index, cla, ' '.join(sent)]
		main_list.append(sub_list)
		index += 1


with open("cluster_text.csv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["ID", "Class", "Text"])
    writer.writerows(main_list)




