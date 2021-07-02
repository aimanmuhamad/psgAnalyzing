#Import the Library
import requests
import re
import time
import datetime
import json
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import patches
import pandas as pd
import parser
import random
from collections import Counter
from bs4 import BeautifulSoup
from sklearn import preprocessing
import urllib
import matplotlib.pyplot as plt
import imp
imp.reload(parser)
%matplotlib inline

team_searched = 'Monaco'
team = {}
team_searched = urllib.parse.quote(team_searched.encode('utf-8'))
search_link = "http://us.soccerway.com/search/teams/?q={}".format(team_searched)
response = requests.get(search_link)
bs = BeautifulSoup(response.text, 'lxml')
results = bs.find("ul", class_='search-results')
# Take the first results
link = "http://us.soccerway.com" + results.find_all('a')[0]['href']
print(link)
team['id_'] = results.find_all('a')[0]["href"].split('/')[4]
team['name'] = results.find_all('a')[0].text

#Parsing
nb_pages = 12
games = []
for page_number in range(nb_pages):
    link_base = 'http://us.soccerway.com/a/block_team_matches?block_id=page_team_1_block_team_matches_3&callback_params=' 
    link_ = urllib.parse.quote('{"page":0,"bookmaker_urls":[],"block_service_id":"team_matches_block_teammatches","team_id":%s,\
    "competition_id":0,"filter":"all","new_design":false}' % team['id_']) + '&action=changePage&params=' + urllib.parse.quote('{"page":-%s}' % (page_number))
    link = link_base + link_
    response = requests.get(link)

    test = json.loads(response.text)['commands'][0]['parameters']['content']
    bs = BeautifulSoup(test, 'lxml')

    for kind in ['even', 'odd']:
        for elem in bs.find_all('tr', class_ = kind):
            game = {}
            game["date"] = elem.find('td', {'class': ["full-date"]}).text
            game["competition"] = elem.find('td', {'class': ["competition"]}).text
            game["team_a"] = elem.find('td', class_='team-a').text
            game["team_b"] = elem.find('td', class_='team-b').text
            game['link'] = "http://us.soccerway.com" + elem.find('td', class_='score-time').find('a')['href']
            game["score"] = elem.find('td', class_='score-time').text.replace(' ','')
            if 'E' in game["score"]:
                game["score"] = game['score'].replace('E','')
                game['extra_time'] = True
            if 'P' in game["score"]:
                game["score"] = game['score'].replace('P','')
                game['penalties'] = True
            if datetime.datetime.strptime(game["date"], '%d/%m/%y') < datetime.datetime.now():
                game = parser.get_score_details(game, team)
                time.sleep(random.uniform(0, 0.25))
                game.update(parser.get_goals(game['link']))
            else:
                del game['score']
            games.append(game)
    games = sorted(games, key=lambda x:datetime.datetime.strptime(x['date'], '%d/%m/%y'))

#Analysis Data
df = pd.DataFrame(games)
df.date = pd.to_datetime(df.date.apply(lambda x:'/'.join([x.split('/')[1],x.split('/')[0], x.split('/')[2]])))
df['month'] = df.date.apply(lambda x:x.month)
df['year'] = df.date.apply(lambda x:x.year)
df = df.sort_values('date', ascending=False)
# le = preprocessing.LabelEncoder()
# df.competition = le.fit_transform(df.competition)
df = df[df.date < datetime.datetime.now()]
df['opponent'] = df.apply(lambda x:(x['team_a']+x['team_b']).replace('PSG', ''), axis=1)
df[['competition', 'date', 'result', 'team_a', 'score', 'team_b', 'players_teams_a', 'players_team_b']].head()
cols = ['Corners', 'Fouls', 'Offsides', 'Shots on target', 'Shots wide']
df = df.apply(lambda x:parser.shot_team(x, cols),axis=1)
df = df.drop(cols, axis=1)
df.head(3)

f, ax = plt.subplots(ncols=3, nrows=3, figsize=(20,10))

d = df.competition.value_counts()
sns.barplot(d.index, d, ax=ax[0][0])
ax[0][0].set_title('Competition distribution')

d = {'Scored':df.nb_goals_PSG.mean(), 'Conceided':df.nb_goals_adv.mean()}
sns.barplot(list(d.keys()), list(d.values()) , ax=ax[1][0])
ax[1][0].set_title('Mean scores')

d = sorted(Counter(np.concatenate(list(df.players_teams_a) + list(df.players_team_b))).items(), key=lambda x:x[1], reverse=True)[:6]
sns.barplot([elem[0] for elem in d], [elem[1] for elem in d], ax=ax[2][0])
ax[2][0].set_title('Most occurence in lineup')

d = df.month.value_counts()
sns.barplot(d.index, d, ax=ax[0][1])
ax[0][1].set_title('Monthly distribution')

d = {'Shots':df.shots_on_target_PSG.mean(), 'Conceided':df.shots_on_target_adv.mean()}
sns.barplot(list(d.keys()), list(d.values()) ,ax=ax[1][1])
ax[1][1].set_title('Shots on target')

d = sorted(Counter(np.concatenate(list(df.subs_in_a) + list(df.subs_in_b))).items(), key=lambda x:x[1], reverse=True)[:6]
sns.barplot([elem[0] for elem in d], [elem[1] for elem in d], ax=ax[2][1])
ax[2][1].set_title('Most occurence as a sub')

d = df['opponent'].value_counts()[:7]
plot = sns.barplot(d.index, d, ax=ax[0][2])
_ = plot.set_xticklabels([elem[:12] for elem in d.index], rotation=15)
ax[0][2].set_title('Opponent distribution')

d = sorted(Counter([elem['player'] for elem in np.concatenate([elem for elem in list(df.goals_a) + list(df.goals_b) if type(elem) == list ])]).items(),
            key=lambda x:x[1], reverse=True)[:6]
plot = sns.barplot([elem[0] for elem in d], [elem[1] for elem in d], ax=ax[1][2])
_ = plot.set_xticklabels([elem[0] for elem in d], rotation=15)
ax[1][2].set_title('Goal scorers')

d = sorted(Counter([elem['assist'] for elem in np.concatenate([elem for elem in list(df.goals_a) + list(df.goals_b) if type(elem) == list ]) if 'assist' in elem]).items(), 
           key=lambda x:x[1], reverse=True)[:6]
plot = sns.barplot([elem[0] for elem in d], [elem[1] for elem in d], ax=ax[2][2])
_ = plot.set_xticklabels([elem[0] for elem in d], rotation=15)
ax[2][2].set_title('Assist player')

## Choose competition
league = 'LI1'

df2 = df[df.competition == league]
p = df2.groupby('opponent')[['nb_goals_adv', 'nb_goals_PSG', 'shots_on_target_PSG', 'shots_on_target_adv', 'fouls_PSG', 'fouls_adv']]#.mean().sort_values('nb_goals_PSG', ascending=False)
g_nb = df2.groupby('opponent')[['year']].aggregate(len).rename(columns={'year':'nb_matches'})
g_all = df2.groupby('opponent')[['nb_goals_adv', 'nb_goals_PSG', 'shots_on_target_PSG', 'shots_on_target_adv', 'fouls_PSG', 'fouls_adv']].mean()
opponents = pd.concat([g_nb, g_all], axis=1).sort_values('nb_matches', ascending=False)

map_vic = {}
for team in opponents.index:
    map_vic[team] = df2[df2.opponent == team].result.value_counts().to_dict()
    
opponents = opponents.reset_index()
opponents = pd.concat([opponents, opponents.opponent.map(map_vic).apply(pd.Series).fillna(0)], axis=1)
opponents['%win'] = (opponents['WIN'] / opponents['nb_matches']).apply(lambda x:round(100*x,1))
opponents.head(10)

## Choose competition
league = 'LI1'

df2 = df[df.competition == league]
p = df2.groupby('opponent')[['nb_goals_adv', 'nb_goals_PSG', 'shots_on_target_PSG', 'shots_on_target_adv', 'fouls_PSG', 'fouls_adv']]#.mean().sort_values('nb_goals_PSG', ascending=False)
g_nb = df2.groupby('opponent')[['year']].aggregate(len).rename(columns={'year':'nb_matches'})
g_all = df2.groupby('opponent')[['nb_goals_adv', 'nb_goals_PSG', 'shots_on_target_PSG', 'shots_on_target_adv', 'fouls_PSG', 'fouls_adv']].mean()
opponents = pd.concat([g_nb, g_all], axis=1).sort_values('nb_matches', ascending=False)

map_vic = {}
for team in opponents.index:
    map_vic[team] = df2[df2.opponent == team].result.value_counts().to_dict()
    
opponents = opponents.reset_index()
opponents = pd.concat([opponents, opponents.opponent.map(map_vic).apply(pd.Series).fillna(0)], axis=1)
opponents['%win'] = (opponents['WIN'] / opponents['nb_matches']).apply(lambda x:round(100*x,1))
opponents.head(10)

def change_name_col(row):
    if row["team_a"] == 'PSG':
        row['goals_PSG'] = row['goals_a']
        row['goals_adv'] = row['goals_b']
        row['lineup_PSG'] = row['players_teams_a']
        row['lineup_adv'] = row['players_team_b']
        row['subs_PSG'] = row['subs_in_a']
        row['subs_adv'] = row['subs_in_b']
    else:    
        row['goals_PSG'] = row['goals_b']
        row['goals_adv'] = row['goals_a']
        row['lineup_PSG'] = row['players_team_b']
        row['lineup_adv'] = row['players_teams_a']  
        row['subs_PSG'] = row['subs_in_a']
        row['subs_adv'] = row['subs_in_b']
    del row['goals_a'], row['goals_b'], row['players_team_b'], row['players_teams_a'], row['subs_in_b'], row['subs_in_a']
    return row

df2 = df.apply(change_name_col, axis=1)

## Choose opponent
opponent = 'Olympique Mars…'

df3 = df2[df2.opponent == opponent]

temp = df3[['lineup_PSG']].rename(columns = {'lineup_PSG': 'name'})
temp['team'] = 'PSG'
temp2 = df3[['lineup_adv']].rename(columns = {'lineup_adv': 'name'})
temp2['team'] = opponent
d = pd.concat([temp, temp2])
rows = []
_ = d.apply(lambda row: [rows.append([nn, row['team']]) 
                         for nn in row["name"]], axis=1)
df_new = pd.DataFrame(rows, columns=d.columns)

f,ax = plt.subplots(nrows=3, figsize=(20,8))
d = sorted(Counter([elem['player'] for elem in np.concatenate(list(df3.goals_PSG+df3.goals_adv))]).items(), key=lambda x:x[1], reverse=True)[:10]
sns.barplot([elem[0] for elem in d], [elem[1] for elem in d], palette=colors, ax =ax[0])
_ = ax[0].legend(handles=[patches.Patch(color=C, label=L) for L, C in tuple(map_color.items())])

d = sorted(Counter([elem['assist'] for elem in np.concatenate(list(df3.goals_PSG+df3.goals_adv)) if 'assist' in elem]).items(), key=lambda x:x[1], reverse=True)[:10]
sns.barplot([elem[0] for elem in d], [elem[1] for elem in d], palette=colors, ax =ax[1])
_ = ax[1].legend(handles=[patches.Patch(color=C, label=L) for L, C in tuple(map_color.items())])

d = df3.result.value_counts().to_dict()
sns.barplot(list(d.keys()), list(d.values()), ax =ax[2])

map_color = {'PSG':'#d1043a', 'Olympique Mars…':'#00aae2'}
colors = {k:map_color[v] for k,v in df_new[['name','team']].set_index('name').to_dict()['team'].items()}

plt.figure(figsize=(15,6))
d = df_new.name.value_counts()[:10]
p = sns.barplot(d.index, d, palette=colors)
legend_patches = [matplotlib.patches.Patch(color=C, label=L) for C, L in zip([item.get_facecolor() for item in Boxes], legend_labels)]
plt.legend(handles=legend_patches)

seasons = {
    "2016/2017" : "12611",
    "2015/2016" : "11645",
    "2014/2015" : "9771",
    "2013/2014" : "8463",
    "2012/2013" : "7242",
    "2011/2012" : "5962",
    "2010/2011" : "5085",
    "2009/2010" : "3455"
}

def convert_int(string):
    try:
        string = int(string)
        return string
    except:
        return string

team['squad'] = {}
for k,v in seasons.items():
    link_base = 'http://us.soccerway.com/a/block_team_squad?block_id=page_team_1_block_team_squad_3&callback_params='
    link_ = urllib.parse.quote('{"team_id":%s}' % team['id_']) + '&action=changeSquadSeason&params=' + urllib.parse.quote('{"season_id":%s}' % v)
    link = link_base + link_
    
    response = requests.get(link)
    test = json.loads(response.text)['commands'][0]['parameters']['content']
    bs = BeautifulSoup(test, 'lxml')
    
    players = bs.find('tbody').find_all('tr')
    squad = [{
        k: convert_int(player.find('td', class_=k).text)
          for k in [k for k,v in Counter(np.concatenate([elem.attrs['class'] for elem in player.find_all('td')])).items() 
                    if v < 2 and k not in ['photo', '', 'flag']]
     } for player in players]
    team['squad'][k] = squad
    try:
        coach = {'position': 'Coach', 'name':bs.find_all('tbody')[1].text}
        team['coach'][k] = coach
    except: pass

d = pd.DataFrame(list(np.concatenate(list(team['squad'].values()))))
cols = [elem for elem in d if elem not in ['age', 'position', 'shirtnumber']]
stats_all_time = d.groupby('name')[cols].sum()
stats_all_time['goals/games'] = stats_all_time['goals'] / stats_all_time['appearances']
stats_all_time = stats_all_time.reset_index()
stats_all_time['position'] = stats_all_time['name'].map(d[['position', 'name']].set_index('name').to_dict()['position'])
stats_all_time['assists/games'] = stats_all_time['assists'] / stats_all_time['appearances']
stats_all_time['yellow-cards/games'] = stats_all_time['yellow-cards'] / stats_all_time['appearances']
stats_all_time['decives_actions/games'] = (stats_all_time['assists'] + stats_all_time['goals'])/ stats_all_time['appearances']
stats_all_time['min/games'] = stats_all_time['game-minutes'] / stats_all_time['appearances']
stats_all_time = stats_all_time.set_index("name", drop=True)
stats_all_time.sort_values('appearances', ascending=False).head(10)

f, ax = plt.subplots(nrows=4, figsize=(20,10))
plt.subplots_adjust(top=1.5)


for i, kind in enumerate(['assists', 'goals', 'decives_actions', 'yellow-cards']):
    nb_appar = 20
    d1 = stats_all_time[stats_all_time.appearances > nb_appar].sort_values('{}/games'.format(kind), ascending=False)
    d = d1['{}/games'.format(kind)][:10]
    sns.barplot(d.index, d, ax=ax[i])
    ax[i].set_title('{} per game since 2009/2010 with more than {} appearances'.format(kind.title(), nb_appar))
    for k, p in enumerate(ax[i].patches):
        ax[i].annotate('%.2f (%d games)' % (p.get_height(), d1.appearances[k]), (p.get_x() * 1.01, p.get_height() * 1.01))

with open('./teams/psg.json', 'r') as f:
    team = json.load(f)
df = parser.convert_df_games(team)

goal_ratio, assist_ratio, decisive_ratio = parser.ratio_one_opponent(df, "Saint-Étienne", "psg")
df_oppo = df[df.opponent == "Saint-Étienne"]

stats_player = parser.player_per_opponent(df, 'E. Cavani', "psg")
fav_opponents = sorted([(k,v) for k,v in stats_player.items() if v["games"] > 2], key=lambda x:x[1]["goals_game"], reverse=True)

f, ax = plt.subplots(ncols=2, nrows=3, figsize=(17,9))
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.2, wspace=None, hspace=None)
plot = sns.barplot([e[0] for e in goal_ratio], [e[1] for e in goal_ratio], ax = ax[0][0])
_ = plot.set_xticklabels([e[0] for e in goal_ratio], rotation=15)
_ = ax[0][0].set_title('Goal/game')

plot = sns.barplot([e[0] for e in decisive_ratio], [e[1] for e in decisive_ratio], ax = ax[0][1])
_ = plot.set_xticklabels([e[0] for e in decisive_ratio], rotation=15)
_ = ax[0][1].set_title('Decisive plays/game')
_ = ax[0][1].set_ylabel("Goals + assist / game")

plot = sns.barplot(y=[df_oppo.nb_goals_PSG.mean(), df_oppo.nb_goals_adv.mean()], x = ['PSG', 'St-Étienne'], ax= ax[1][0])
_ = ax[1][0].set_title('Goal/game on average')

plot = sns.barplot([e[0] for e in fav_opponents][:8], [e[1]["goals_game"] for e in fav_opponents][:8], ax= ax[1][1])
_ = ax[1][1].set_title('%s\'s favorite opponent' % player)
_ = ax[1][1].set_ylabel("Goals")
for k, p in enumerate(ax[1][1].patches):
    ax[1][1].annotate('%d (%d games)' % (fav_opponents[k][1]["goals"], fav_opponents[k][1]["games"]), (p.get_x(), p.get_height()+0.02))
    
d = df_oppo[df_oppo.place == "away"]
d2 = d.result.value_counts()
sns.barplot(d2.index, d2, ax= ax[2][0])
_ = ax[2][0].set_title('Results for away games')

d = df_oppo[df_oppo.place == "away"]
d2 = d.score.value_counts()
sns.barplot(d2.index, d2, ax= ax[2][1])
_ = ax[2][1].set_title('Scores for away games')
f.suptitle("%s against %s" % (team['name'], opponent), x=0.5, y=0.05)