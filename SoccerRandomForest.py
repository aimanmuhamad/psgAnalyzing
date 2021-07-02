#Import the Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import imp
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVC
from sklearn.model_selection import train_test_split
import whoscored
from unidecode import unidecode
import pickle
import difflib
import glob
%matplotlib inline

#Data Processing
teams = {}
for file in glob.glob('./games/*.json'):
    with open(file,'r') as f:
        teams[file.split('/')[2].split('_')[0]] = json.load(f)

def get_position_sub(row):
    if row['Position'] == 'Sub':
        name = row['Player']
        try:
            row['Position'] = [k for k in df[df.Player == name].Position.value_counts().index if k !='Sub'][0]
        except:
            row['Position'] = 'substitute'
    return row

stop_words = ['<','Bl.', 'Exc.', '>']
def mean_rating(row):
    if pd.isnull(row['Rating_LEquipe']) or row['Rating_LEquipe'] in stop_words:
        name = row['Player']
        temp = df[(df.Player == name) & (~df.Rating_LEquipe.isin(stop_words)) & (pd.notnull(df.Rating_LEquipe))]
        if len(temp) > 4:
            row['Rating_LEquipe'] = np.mean(temp.Rating_LEquipe.apply(int))
    return row

position_mapping = {'attackingmidfieldcenter' : 'attackingmidfield',
 'attackingmidfieldleft' : 'attackingmidfield',
 'attackingmidfieldright' : 'attackingmidfield',
 'defenderleft' : 'defenderlateral',
 'defendermidfieldcenter' : "defendermidfield",
 'defendermidfieldleft' : 'defendermidfield',
 'defendermidfieldright': "defendermidfield",
 'defenderright': 'defenderlateral',
 'forwardleft' : 'forwardlateral',
 'forwardright' : 'forwardlateral',
 'midfieldcenter' :'midfield' ,
 'midfieldleft' :'midfield' ,
 'midfieldright' :'midfield' }

mapping_team_name = {'ASM': 'Monaco',
 'ASNL': 'Nancy',
 'ASSE': 'Saint-Etienne',
 'DFCO': 'Dijon',
 'EAG': 'Guingamp',
 'FCGB': 'Bordeaux',
 'FCL': 'Lorient',
 'FCM': 'Metz',
 'FCN': 'Nantes',
 'Losc': 'Lille',
 'MHSC': 'Montpellier',
 'Man. City': 'Manchester City',
 'Man. United': 'Manchester United',
 'OGCN': 'Nice',
 'OL': 'Lyon',
 'OM': 'Marseille',
 'PSG': 'Paris Saint Germain',
 'Palace': 'Crystal Palace',
 'SCB': 'SC Bastia',
 'SCO': 'Angers',
 'SMC': 'Caen',
 'SRFC': 'Rennes',
 'Stoke City': 'Stoke',
 'TFC': 'Toulouse',
 'WBA': 'West Bromwich Albion'}

df = pd.DataFrame()
for team, file in teams.items():
    for i, game in enumerate(file):
        temp = pd.DataFrame(game['stats'])
        temp['Opponent'] = game['opponent']
        temp['Place'] = game['place'].title()
        result = game['result'].split(' : ')
        temp['Goal Team'] = int(result[int(game['place'] == 'away')])
        temp['Goal Opponent'] = int(result[int(game['place'] == 'home')])
        temp['Team'] = {'psg':'Paris Saint Germain'}.get(team, team).title()
        temp['Team'] = {'Bastia': 'SC Bastia'}.get(temp['Team'][0], temp['Team'][0])
        temp['Day'] = i+1
        df = pd.concat([df, temp])
df = df.apply(whoscored.get_name, axis=1).reset_index(drop=True)
df['LineUp'] = 1
df.loc[df.Position == 'Sub', 'LineUp'] = 0
df = df.apply(get_position_sub, axis=1)
df.Goal.fillna(0, inplace=True)
df.Assist.fillna(0, inplace=True)
df.Yellowcard.fillna(0, inplace=True)
df.Redcard.fillna(0, inplace=True)
df.Penaltymissed.fillna(0, inplace=True)
df.Shotonpost.fillna(0, inplace=True)

with open('./ratings/notes_ligue1_lequipe.json','r') as f:
    rating_lequipe  = json.load(f)
rating_lequipe = {mapping_team_name.get(k,k):[p for p in v if list(p.keys())[0] != 'Nom' and len(list(p.values())[0]) > 0] 
                    for k,v in rating_lequipe.items()}

for team_name, rating_team in rating_lequipe.items():
    if team_name in set(df.Team):
        players_lequipe = [list(k.keys())[0] for k in rating_team]
        players_df = list(set(df[df.Team == team_name].Player))
        for player in rating_team:
            [(player_name, player_ratings)] = player.items()
            try:
                player_name_df = difflib.get_close_matches(player_name, players_df)[0]
            except:
                if len(unidecode(player_name).split('-')) > 1 :
                    player_name_df = [k for k in players_df if unidecode(player_name).split('-')[0].replace("'","").lower() in unidecode(k).replace("'","").lower()
                                                             or unidecode(player_name).split('-')[1].replace("'","").lower() in unidecode(k).replace("'","").lower()][0]
                else:
                    player_name_df = [k for k in players_df if unidecode(player_name).replace("'","").lower() in unidecode(k).replace("'","").lower()][0]
            for day, rating in player_ratings.items():
                df.loc[(df.Player == player_name_df) & (df.Team == team_name) & (df.Day == int(day.split('Day ')[1])), 'Rating_LEquipe'] = rating
df = df.apply(mean_rating, axis=1)
df.drop('null', axis=1, inplace=True)

print('%d/%d données avec une note l\'Équipe' % (len(df[(~df.Rating_LEquipe.isin(stop_words)) & (pd.notnull(df.Rating_LEquipe))]),
                                                len(df)))
df[['Team','Goal Team', 'Goal Opponent', 'Player', 'Goal',
    'Opponent', 'Day', 'Position', 'Age', 'LineUp', 'Rating_LEquipe']].sort_values('Player').sample(8)

with open('./df.pkl', 'rb') as f:
    df = pickle.load(f)

other_cols = ['Day', "Opponent", "Place", "Player", "Position", 
                "Team", 'Key Events', "Rating_LEquipe"]
col_delete = ['Key Events', "Rating_LEquipe", 'Rating', 'Day']
cols_to_transf = [col for col in df.columns if col not in other_cols]
p = pd.concat([df[cols_to_transf].applymap(float), df[other_cols]], axis=1)
new_col_to_remov = [e for e in p.columns if e.startswith('Acc')] + ['Touches']

p = p[(~p.Rating_LEquipe.isin(stop_words)) & (pd.notnull(p.Rating_LEquipe))].set_index('Player')
X = p[[col for col in p.columns if col not in col_delete + new_col_to_remov]]
y = p['Rating_LEquipe'].apply(float)

label_encoders = {}
col_encode = ["Opponent", "Place", "Position", "Team"]
for col in col_encode:
    label_encoders[col.lower()] = LabelEncoder()
    X[col] = label_encoders[col.lower()].fit_transform(X[col])

d = []
k_fold = 5
for k in range(k_fold): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/k_fold)
    ridge = Ridge(alpha=0.001, max_iter=10000, tol=0.0001)
    ridge.fit(X_train, y_train)
    d.append(np.mean((ridge.predict(X_test) - y_test) ** 2))
print('%d-fold cross validation mean square error : %.2f' % (k_fold, np.mean(d)))
ridge = Ridge(alpha=0.001, max_iter=10000, tol=0.0001)
_ = ridge.fit(X, y)

top_k = 10
l = np.argsort(list(map(np.abs, ridge.coef_)))[::-1][:top_k]
print("The %d most important parameters are:\n%s" % (top_k,'\n'.join(['%s : %.2f' % (a,b) 
                                                                       for a,b in zip(X_train.columns[l], ridge.coef_[l])])))

rf = RandomForestRegressor(oob_score=True, n_estimators=100)
rf.fit(X, y)
print('RF : oob score : %.2f' % rf.oob_score)
d = []
k_fold = 5
for k in range(k_fold): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/k_fold)
    rf = RandomForestRegressor(oob_score=True, n_estimators=100)
    rf.fit(X_train, y_train)
    d.append(np.mean((rf.predict(X_test) - y_test) ** 2))
print('RF : %d-fold cross validation mean square error : %.2f' % (k_fold, np.mean(d)))