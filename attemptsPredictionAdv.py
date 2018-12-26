# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 08:19:26 2018

@author: Hrishikesh S
@link : https://github.com/redlegblackarm/ProblemRecommenderSystem
Instructions : documentation string used for explanantion
			   '#' used for commenting code
			   The code will take a lot of time to execute
Explanation : I am creating a profile for each user. Each profile has 14 levels, which corresponds to the 14 types of problem difficulties.
			  So we are going to add the attempts range for each profile, by taking data from 'user_data.csv' and then divide it by the number of counts.
			  While testing, depending on the user and the problem we are solving, using the problem difficulty, we'll choose the required profile 
			  with the corresponding profile. If the profile-level value is not zero, then that is the predicted attempts-range and it will be written into 
			  'test_predictions.csv'. If the profile-level value is zero, we'll the choose the next 3 most similar users' attempts-range. This is useful
			  when users have tried only problems of a particular level/category and we are using KNN(k-nearest neighbours) for determining most similar users
			  for every user. In my implementation, I am using only 3 most similar users. To make the model more accurate, I have to implement KNN for k>3 
			  as there are lesser zeroes as k value increases. 
			  Incase, all similar users have profile-level value as 0, then we'll give the median attempts-range value of 4 to that problem, for that user.
Future of the project : 1. Increase k-value for KNN
						2. using another model(like random forests) for this problem
"""
# cd "Desktop/Third Year/Data Analytics/Project Ideas/Final Project"

import csv 
from sklearn.metrics import mean_squared_error as rmse
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

np.seterr(divide='ignore', invalid='ignore')

#def rmse(y,y_pred):
#    y = [(x-y_pred)**2 for x in y]
#    mse = np.mean(y)
#    return math.sqrt(mse)

def convert(data):
    rows = data.user_id.unique()
    cols = data.problem_id.unique()
    #print(len(rows), len(cols))
    data = data[['user_id', 'problem_id', 'attempts_range']]
    idict = dict(zip(cols, range(len(cols))))
    udict = dict(zip(rows, range(len(rows))))
    data.user_id = [ udict[i] for i in data.user_id ] 
    data['problem_id'] = [ idict[i] for i in data['problem_id'] ]
    nmat = data.as_matrix()
    return nmat, len(rows), len(cols)

def loc(mat, n_rows, n_cols):
    naive = np.zeros((n_rows, n_cols))
    print('naive shape', naive.shape)
    for row in mat:
        naive[int(row[0]), int(row[1])] = row[2]
    amean1 = np.mean(naive[naive!=0])
    umean1 = sum(naive.T) / sum((naive!=0).T)
    imean1 = sum(naive) / sum((naive!=0))
    return naive, amean1, umean1, imean1

def cos(mat, a, b):
    if a == b:
        return 1
    aval = mat.T[a].nonzero()
    bval = mat.T[b].nonzero()
    corated = np.intersect1d(aval, bval)
    if len(corated) == 0:
        return 0
    avec = np.take(mat.T[a], corated)
    bvec = np.take(mat.T[b], corated)
    val = 1 - cosine(avec, bvec)
    if np.isnan(val):
        return 0
    return val

def usersimilar(mat):
    # *Calculate amean, umean and imean as before
    n = mat.shape[1]
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_mat[i][j] = cos(mat, i, j)
    return sim_mat

def kmostsimilarusers(sim_mat, k):
    ksimilar = []
    for ii, i in enumerate(sim_mat):
        mi = i.argsort()
        mi = np.delete(mi, np.where(mi==ii))
        ksimilar.append(mi[-(k):])
    return ksimilar

def cos_inverse(mat, a, b):
    if a == b:
        return 1
    aval = mat.T[a].nonzero()
    bval = mat.T[b].nonzero()
    corated = np.intersect1d(aval, bval)
    if len(corated) == 0:
        return 0
    avec = np.take(mat.T[a], corated)
    bvec = np.take(mat.T[b], corated)
    val = cosine(avec, bvec)
    if np.isnan(val):
        return 0
    return val

def itemssimilar(mat):
    # *Calculate amean, umean and imean as before
    n = mat.shape[1]
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_mat[i][j] = cos_inverse(mat, i, j)
    return sim_mat
"""KNN"""
def mostSimilarUser():
    train_matrix, train_rows, train_cols = convert(complete_user_data)
    print(train_matrix, train_rows, train_cols)
    train_matrix = train_matrix[~np.isnan(train_matrix).any(axis=1)]
    train_naive, train_amean1, train_umean1, train_imean1 = loc(train_matrix, train_rows, train_cols)
    train_usim_matrix_cosine = usersimilar(train_naive.T)
    k = 3    
    train_ksimilar_cosine = kmostsimilarusers(train_usim_matrix_cosine, k)
    #train_udisim_matrix_cosine = userdissimilar(train_naive.T) 
    #train_kdisimilar_cosine = kmostsimilarusers(train_udisim_matrix_cosine, k)    
    #number of problems user wants
    #number_of_problems = int(input())
    #recommended_problem = []
    #for i in range(len(train_ksimilar_cosine)):
    #    for j in range(k):
    #        sim_user = all_users[train_ksimilar_cosine[i][j]]
    #        curr_user = all_users[i]
    #        sim_problem = complete_user_data[complete_user_data['user_id'] == sim_user]['problem_id']
    #        curr_problem = complete_user_data[complete_user_data['user_id'] == curr_user]['problem_id']
    #        r = [item for item in sim_problem if item not in curr_problem]
    #        recommended_problem.append(r[0:number_of_problems])
    return train_ksimilar_cosine
    
"""Reading the files"""
problem_data = pd.read_csv("train/problem_data.csv")
print(problem_data.head())
problem_data = problem_data.fillna('A')

train_subs = pd.read_csv("train/train_submissions.csv")
print(train_subs.head())
train_subs = train_subs.fillna('A')

user_data = pd.read_csv("train/user_data.csv")
print(user_data.head())

complete_user_data = pd.merge(user_data, train_subs, on="user_id", how="outer")
print(complete_user_data.head())

all_users = user_data['user_id']

"""listing all similar users which will be used for new users"""
similar_users = mostSimilarUser()

problem_data = []
train_subs = []

with open("train/problem_data.csv") as fproblem:
    reader = csv.reader(fproblem)
    for line in reader:
        problem_data.append(line)
    problem_data = problem_data[1:]

with open("train/train_submissions.csv") as ftrain:
    reader = csv.reader(ftrain)
    for line in reader:
        train_subs.append(line)
    train_subs = train_subs[1:]

"""For missing problem ranking, we'll put A"""
for i in problem_data:
    if i[1] == '':
        i[1] = 'A'

"""reading test file"""
test = pd.read_csv("test/test.csv")

"""unique users extraction"""
unique_users = list(user_data['user_id'])

"""profile creation"""
profile = [[0]*14 for _ in unique_users]
count = [[0]*14 for _ in unique_users]

#for u, user in enumerate(unique_users):
#    i = 0
#    print(user)
#    for i in range(0, train_subs.shape[0]):
#        if train_subs['user_id'][i] == user:
#            j = 0
#            for problem in problem_data['problem_id']:
#                if(problem == train_subs['problem_id'][i]):
#                    print(train_subs['user_id'][i], user)
#                    print(problem, train_subs['problem_id'][i])
#                    print(u, problem_data['level_type'][j]) 
#                    print(ord(problem_data['level_type'][j]) - 65, 
#                          train_subs['attempts_range'][i])
#                    profile[u][ord(problem_data['level_type'][j]) - 65] += train_subs['attempts_range'][i]
#                    count[u][ord(problem_data['level_type'][j]) - 65] += 1
#                else:
#                    j = j + 1

#for u, user in enumerate(profile):
#    for l, level in enumerate(user):
#        if count[u][l] != 0:
#            profile[u][l] = round(level / count[u][l])

for u, user in enumerate(unique_users):
    for subs in train_subs:
        if subs[0] == user:
            for problem in problem_data:
                if problem[0] == subs[1]:
                    #print(u, ord(problem[1]) - 65)
                    profile[u][ord(problem[1]) - 65] += int(subs[2])
                    count[u][ord(problem[1]) - 65] += 1
                    break
                    
for u, user in enumerate(profile):
    for l, level in enumerate(user):
        if count[u][l] != 0:
            profile[u][l] = round(level / count[u][l])

"""predicting the ratings and doing in sample validation"""
#pred = []
#actual = []

#j = 0
#for subs in train_subs['problem_id']:
#    i = 0
#    flag =  False
#    for problem in problem_data['problem_id']:
#        if problem == subs:
#            print(problem, subs)
#            query_level = ord(problem_data['level_type'][i]) - 65
#            flag = True
#            break
#        i = i + 1
#    if(flag):
#        print(subs, train_subs['user_id'][j])
#        print(problem_data['level_type'][i], query_level, j)
#        a = train_subs[train_subs['problem_id'] == subs]
#        print(a)
#        print(a[a['user_id'] == train_subs['user_id'][j]])
#        print("___________________________________________________")
#        pred.append(profile[unique_users[unique_users == train_subs['user_id'][j]].index[0]][query_level])
#        actual.append(train_subs[train_subs['problem_id'] == subs][train_subs['user_id'] == train_subs['user_id'][j]])
#        j = 0
#        flag = False
#    else:
#        j = j + 1

#print('rmse for our prediction(using in sample validation) :', end=' ')
#print(rmse(actual, pred))

#predicting the ratings and doing in sample validation
pred = []
actual = []

for subs in train_subs:
    for p, problem in enumerate(problem_data):
        if problem[0] == subs[1]:
            query_level = ord(problem[1]) - 65
            break
    pred.append(profile[unique_users.index(subs[0])][query_level])
    actual.append(int(subs[2]))
    
print('rmse for our prediction(using in sample validation) :', end=' ')
print(rmse(actual, pred))

"""prediciton for user and problem queried"""
query_user = 'user_' + input('user_id(number) : ')
query_problem = 'prob_' + input('prob_id(number) : ')

for p, problem in enumerate(problem_data):
    if problem[0] == query_problem:
        query_level = ord(problem[1]) - 65
        break
print('prediction by our system :', end=' ')
print(profile[unique_users.index(query_user)][query_level])

"""working on test data"""
with open("test/test_submissions.csv", "w", newline = '') as f:
    fwriter = csv.writer(f)
    missing_values = 0
    #missing_similar_users = 0
    for i in range(0, len(test['user_id'])):
        try:
            query_user = test['user_id'][i]
            query_problem = test['problem_id'][i]
            for p, problem in enumerate(problem_data):
                if problem[0] == query_problem:
                    query_level = ord(problem[1]) - 65
                    break;
            print('prediction by our system :', end=' ')
            a = [test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(query_user)][query_level]]
            #print(list(test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(query_user)][query_level]))
            print(a)
			"""if the user never attempted to solve a problem of a particular category, we will give the attempts_range of the first most similar user"""
            if(a[1] == 0):
                similar_index = user_data[user_data['user_id'] == query_user].index[0]
                similar_users_for_given_index = similar_users[similar_index]
                print(similar_users_for_given_index)
                #try:
                new_user = user_data['user_id'][similar_users_for_given_index[0]]
                print(new_user)
                print('prediction by our system :', end=' ')
                a = [test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(new_user)][query_level]]
                #print(list(test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(query_user)][query_level]))
                print(a)
				"""if the user never attempted to solve a problem of a particular category, we will give the attempts_range of the second most similar user"""
                if(a[1] == 0):
                    similar_index = user_data[user_data['user_id'] == query_user].index[0]
                    similar_users_for_given_index = similar_users[similar_index]
                    print(similar_users_for_given_index)
                    #try:
                    new_user = user_data['user_id'][similar_users_for_given_index[1]]
                    print(new_user)
                    print('prediction by our system :', end=' ')
                    a = [test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(new_user)][query_level]]
                    #print(list(test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(query_user)][query_level]))
                    print(a)
					"""if the user never attempted to solve a problem of a particular category, we will give the attempts_range of the third most similar user"""
                    if(a[1] == 0):
                        similar_index = user_data[user_data['user_id'] == query_user].index[0]
                        similar_users_for_given_index = similar_users[similar_index]
                        print(similar_users_for_given_index)
                        #try:
                        new_user = user_data['user_id'][similar_users_for_given_index[2]]
                        print(new_user)
                        print('prediction by our system :', end=' ')
                        a = [test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(new_user)][query_level]]
                        #print(list(test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(query_user)][query_level]))
                        print(a)
						"""if all similar users haven't attempted it, we give a median value for the attempts range"""
                        if(a[1] == 0):
                            a[1] = 4
            fwriter.writerow(a)
        except ValueError:
            print('entry is new, need to put average')
            missing_values = missing_values + 1
            similar_index = user_data[user_data['user_id'] == query_user].index[0]
            similar_users_for_given_index = similar_users[similar_index]
            print(similar_users_for_given_index)
            #try:
            new_user = user_data['user_id'][similar_users_for_given_index[0]]
            print(new_user)
            print('prediction by our system :', end=' ')
            a = [test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(new_user)][query_level]]
            #print(list(test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(query_user)][query_level]))
            print(a)
            fwriter.writerow(a)
            #except ValueError:
            #missing_similar_users = missing_similar_users + 1
            #new_user = similar_users_for_given_index[1]
            #print('prediction by our system :', end=' ')
            #a = [test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(new_user)][query_level]]
            #print(list(test['user_id'][i]+"_"+test['problem_id'][i],profile[unique_users.index(query_user)][query_level]))
            #print(a)
            #fwriter.writerow(a)
f.close()
print(missing_values)