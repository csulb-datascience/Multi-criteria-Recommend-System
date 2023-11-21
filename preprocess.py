import pandas as pd


# Load dataset
rating = pd.read_csv('YM/multi_YM.csv')

rating.rename(columns = {'UserID':'user_id', 'ItemID':'item_id'}, inplace = True)

print( "# user:", len(rating["user_id"].unique()))
print( "# item:", len(rating["item_id"].unique()))

# Drop 0 overall ratings
rating = rating.drop(rating[rating["overall"] == 0].index)

# n core settings
core = 5

# 1. count frequencies
users_to_drop= rating["user_id"].value_counts() < core
users_to_drop = users_to_drop.index[users_to_drop].tolist()
rating = rating[~rating['user_id'].isin(users_to_drop)]


n_user = len(rating["user_id"].unique())
n_item = len(rating["item_id"].unique())

print( "# user:", n_user)
print( "# item:", n_item)
print( "# ratings:", len(rating))
print(f"Density: {(len(rating)/(len(rating['user_id'].unique())*len(rating['item_id'].unique()))):.4f}")

# Reordering all
# key : xxx_core // value : [range[0, len(xxx_new)]]
users_core = rating["user_id"].unique().tolist()
items_core = rating["item_id"].unique().tolist()
user_order = [*range(len(users_core))]
item_order = [*range(len(items_core))]

user_id_dict = {users_core[i]: user_order[i] for i in range(len(users_core))}
item_id_dict = {items_core[i]: item_order[i] for i in range(len(items_core))}

# Replace with new orderings
rating = rating.replace({"user_id": user_id_dict})
rating = rating.replace({"item_id": item_id_dict})

rating.to_csv(f'YM/YM_multi_graph.csv', index=None )

# Random split
tr = 0.7
val = 0.15
ts = 0.15

tr_list = []
val_list = []
tval_list = []
ts_list = []

# Suffle ratings
rating = rating.sample(frac=1) 

for u in range(len(rating["user_id"].unique())):
    u_rating = rating[rating['user_id'] == u]

    num_edges = len(u_rating)
    tr_num_edges = int(num_edges * tr)
    val_num_edges = int(num_edges * val)
    ts_num_edges = int(num_edges * ts)

    tr_list.append(u_rating.iloc[:tr_num_edges])
    val_list.append(u_rating.iloc[tr_num_edges:tr_num_edges+val_num_edges])
    ts_list.append(u_rating.iloc[tr_num_edges+val_num_edges:])

tval_list = tr_list+tval_list

tr_rating = pd.concat(tr_list)
val_rating = pd.concat(val_list)
tval_rating = pd.concat(tval_list)
ts_rating = pd.concat(ts_list)

tr_rating.sort_values(by=['user_id'], inplace =True)
val_rating.sort_values(by=['user_id'], inplace =True)
tval_rating.sort_values(by=['user_id'], inplace =True)
ts_rating.sort_values(by=['user_id'], inplace =True)

# MC Expansion graph construction, for training set
# MC Expansion graph is consructed by concatenating dataframe with shifted indices for different criterion-item nodes

n_user = len(rating['user_id'].unique())
n_item = len(rating['item_id'].unique())

# Split rating, and stack
cri_list = list(rating.columns)
cri_list.remove('overall')
cri_list.remove('user_id')
cri_list.remove('item_id')

r1 = tr_rating[['user_id','item_id', 'overall']]

for idx, cri in enumerate(cri_list):
    r2 = tr_rating[['user_id','item_id', cri]]
    r2['item_id'] += n_item * (idx + 1)
    r2 = r2.rename(columns = {cri:'overall'})

    r1 = pd.concat([r1,r2], ignore_index=True, axis=0)
tr_rating_for_test = tr_rating
tr_rating = r1

# For Recbole
import copy

def df2inter(rating, area:str):
    inter_rating = copy.deepcopy(rating)
    inter_rating.rename(columns = {'overall':'rating:float', 'value':'value:float', 'Visuals':'Visuals:float','Direction':'Direction:float','Story':'Story:float','Acting':'Acting:float', "user_id":'user_id:token', "item_id":"item_id:token"}, inplace = True)
    inter_rating.to_csv(f'./YM/YM.{area}.inter', index=False, sep='\t')
    return inter_rating

inter_rating = copy.deepcopy(rating)
inter_rating.rename(columns = {'overall':'rating:float', 'value':'value:float', 'Visuals':'Visuals:float','Direction':'Direction:float','Story':'Story:float','Acting':'Acting:float', "user_id":'user_id:token', "item_id":"item_id:token"}, inplace = True)
inter_rating.to_csv(f'./YM/YM.inter', index=False, sep='\t')


df2inter(tr_rating, "tr")
df2inter(val_rating, "val")
df2inter(ts_rating, "ts")