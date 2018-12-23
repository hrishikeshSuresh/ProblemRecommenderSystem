# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 20:53:55 2018

@author: Hrishikesh S
"""
import csv 
from sklearn.metrics import mean_squared_error as rmse

#file reading
problem_data = []
train_subs = []
with open("dataset/train/problem_data.csv") as fproblem:
    reader = csv.reader(fproblem)
    for line in reader:
        problem_data.append(line)
    problem_data = problem_data[1:]
with open("dataset/train/train_submissions.csv") as ftrain:
    reader = csv.reader(ftrain)
    for line in reader:
        train_subs.append(line)
    train_subs = train_subs[1:]

#imputation of level_type with mode value
for i in problem_data:
    if i[1] == '':
        i[1] = 'A'

#unique users extraction
unique_users = list(set(list(map(list, zip(*(train_subs))))[0]))

#profile creation
profile = [[0]*14 for _ in unique_users]
count = [[0]*14 for _ in unique_users]

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

#prediciton for user and problem queried
query_user = 'user_' + input('user_id(number) : ')
query_problem = 'prob_' + input('prob_id(number) : ')

for p, problem in enumerate(problem_data):
    if problem[0] == query_problem:
        query_level = ord(problem[1]) - 65
        break
print('prediction by our system :', end=' ')
print(profile[unique_users.index(query_user)][query_level])