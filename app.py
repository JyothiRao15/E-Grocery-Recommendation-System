import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

sim = pickle.load(open("similarity.pkl","rb"))
item_matrix=pickle.load(open('item_matrix.pkl','rb'))
user_vecs=pickle.load(open('user_vecs.pkl','rb'))
item_vecs=pickle.load(open('item_vecs.pkl','rb'))
cust_dict=pickle.load(open('cust_dict.pkl','rb'))


data1 = pd.read_csv('item_dummy.csv')
#item_dummy is a dataframe that consists of only items.
#data1.shape rows = 115, cols = 1
item_list=data1.values.tolist()
item_dict={}
counter=0
for i in item_list:
    item_dict[counter]=i
    counter += 1

def rcmd(m,n):
    if m not in data1['item'].unique():
        return('This item is not in our database.\nPlease check if you spelled it correct.')
    else:
        i = data1.loc[data1['item']==m].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:n+1]

        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data1['item'][a])
        return l


purchased_items=[]

def recm(cust_id):
    cust_index=[value for (key,value) in cust_dict.items() if key==cust_id][0]
    purchased_index=item_matrix[cust_index,:].nonzero()[1]
    purchased_items = []
    for i in purchased_index:
        purchased_items.append(item_dict[i])
    return purchased_items


rec_list = []

def rec(cust_id,num_items):
    cust_index=[value for (key,value) in cust_dict.items() if key==cust_id][0]
    pref_vec = item_matrix[cust_index,:].toarray()
    pref_vec = (pref_vec.reshape(-1)) + 1
    pref_vec[pref_vec > 1] = 0
    rec_vector = user_vecs[cust_index,:].dot(item_vecs.T)
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = (pref_vec)*(rec_vector_scaled)
    product_idx = np.argsort(recommend_vector)[::-1][:num_items]
    rec_list = []
    for i in product_idx:
        rec_list.append(item_dict[i])
    return rec_list

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('initial.html')

@app.route("/item")
def recommend():
    item = request.args.get('item')
    number = int(request.args.get("n"))
    r = rcmd(item,number)
    item = item.upper()
    if type(r)==type('string'):
        return render_template('first.html',item=item,r=r,t='s')
    else:
        return render_template('first.html',item=item,r=r,t='l')

@app.route("/p")
def Enter():
    cust_id = request.args.get('cust_id')
    a = recm(cust_id)
    return render_template('p.html',purchased=a)

@app.route("/r")
def Submit():
    cust_id = request.args.get('cust_id')
    number = int(request.args.get("num_items"))
    z = rec(cust_id,number)
    return render_template('r.html',recd=z)

if __name__ == '__main__':
    app.run(debug=True)