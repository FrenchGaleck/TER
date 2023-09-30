import sqlite3
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import time
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

np.random.seed(42)

pd.set_option('display.max_columns', None)
# First, create a connection object that represents the database
conn = sqlite3.connect('/Users/franc/PycharmProjects/TER/database.sqlite')
# Second, create a cursor
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Récupérer les résultats
tables = cur.fetchall()

# Afficher les noms des tables
"""for table in tables:
    print(table[0])"""
# Exécuter la commande SQL pour obtenir la liste des colonnes de la table "ma_table"
cur.execute("PRAGMA table_info(Player_Attributes)")

joueur = pd.read_sql("SELECT * FROM Player;", conn)
# meme chose des ligues
ligue = pd.read_sql("SELECT * FROM League;", conn)

match = pd.read_sql("SELECT * FROM Match;", conn)
team_attribue = pd.read_sql("SELECT * FROM Team_Attributes;", conn)

# je récupère la bdd des joueurs
joueur_attribue = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
# dommage que les stats de tirs pour, contre, possesion sont pas complete, je prefère donc les enlever (seulement 20% des matchs ont ces stats)
# j'enlève également la grosse majorité des colonnes qui ne servent pas à grands chose), suppression de toutes les colonnes contenant
# les cotes des bets car l'objectif c'est de predire à partir de statistiques, pas à partir des cotes des matchs

reduction_match = ['id', 'country_id', 'league_id', 'season', 'stage', 'date', 'match_api_id', 'home_team_api_id',
                   'away_team_api_id'
    , 'home_team_goal', 'away_team_goal', 'home_player_1', 'home_player_2', 'home_player_3', 'home_player_4',
                   'home_player_5', 'home_player_6'
    , 'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                   'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6'
    , 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11','B365H','B365D','B365A']
match = match[reduction_match]
# match.dropna(inplace=True)

match = match[match['country_id'].isin([4769, 7809, 1729, 21518,10257])]#, 7809, 1729, 21518,10257,7809
match.dropna(subset=['B365H', 'B365D', 'B365A'], inplace=True)

championnats = [4769, 7809, 1729, 21518, 10257]  # Identifiants des championnats
"""
matchs_selectionnes = pd.DataFrame()
for league_id in championnats:
    match_championnat = match[match['country_id'] == league_id].tail(1500)
    matchs_selectionnes = pd.concat([matchs_selectionnes, match_championnat], ignore_index=True)
match = matchs_selectionnes# print(match)"""
# objectif créé une nouvelle data frame avec les colonnes suivantes : état de forme de l'équipe avec le nb de victoire sur
# les 10 derniers match, le nombre de match nul, le nombre de défaite ou alors juste une colonne avec le nombre de points pris sur les 0 derniers match
# ensuite un tableau avec la moyenne du rank des joueurs présent sur le terrain
# nombre de point en confrontation direct sur les 5 derniers match
# nombre de but pour, nombre de but contre
"""
listes des fonctions : 
 resultat(match,id_team) : resultat du match a partir de l'id d'une equipe,on l'utilisera pour calculer la forme d'une eq

 diff_form(match,match_id): fera la difference de forme entre 2 equipes

 niv_dom(match),niv_ext(match),diff_niv (match) : fais la moyenne de l'eq a dom, a l'ext et on prends la difference

 but_marque(match,id_team),but_encaisse(match,id_team): permet d'obtenir les but enc et marq d'une eq dans un match

 nombre_but_match(match,match_id): permet d'avoir le nombre de but par match de l'eq a dom marquer,encaissé
                                   et de l'equipe exterieur, renvoie également la difference

"""


# résultat du match afin de calculer les points, sert pour la fonction d'après pour obtenir état de forme des équipes
def resultat(match, id_team):
    but_dom = match.home_team_goal
    but_ext = match.away_team_goal
    # print(but_dom)
    # print(but_ext)
    if (id_team == match.home_team_api_id):
        if (but_dom == but_ext):
            return 1
        if (but_dom > but_ext):
            return 3
        if (but_dom < but_ext):
            return 0
    else:
        if (but_dom == but_ext):
            return 1
        if (but_dom < but_ext):
            return 3
        if (but_dom > but_ext):
            return 0


# du coup val pos, forme en faveur de l'équipe a domicile, val neg opposé
def diff_form(match, match_id):
    date_match = match_id['date']
    equipe_dom_id = match_id.home_team_api_id
    equipe_ext_id = match_id.away_team_api_id
    # print(equipe_dom_id)
    # print(equipe_ext_id)
    derniers_matchs_dom = match[(match.home_team_api_id == equipe_dom_id) | (match.away_team_api_id == equipe_dom_id)]
    derniers_matchs_dom = derniers_matchs_dom[derniers_matchs_dom['date'] < date_match].tail(5)
    points_dom = sum(derniers_matchs_dom.apply(lambda x: resultat(x, equipe_dom_id), axis=1))
    # Calcul des points pour l'équipe à l'extérieur
    derniers_matchs_ext = match[
        (match['home_team_api_id'] == equipe_ext_id) | (match['away_team_api_id'] == equipe_ext_id)]
    derniers_matchs_ext = derniers_matchs_ext[derniers_matchs_ext['date'] < date_match].tail(5)
    # print(derniers_matchs_ext)
    points_ext = sum(derniers_matchs_ext.apply(lambda x: resultat(x, equipe_ext_id), axis=1))
    # print("résultat: ")
    # print(match_id.home_team_goal,match_id.away_team_goal)
    diff_point = points_dom - points_ext
    return diff_point, points_dom, points_ext

joueur_opti = joueur_attribue.groupby('player_api_id')
# cette fonction va faire en sorte qu'on est la moyenne du niveau de l'équipe à domicile
def niv_dom(match):
    date_match = match['date']
    somme = 0
    coef = 1.3
    joueur_stat = pd.DataFrame()
    match_id = match.match_api_id
    home_team = ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6'
        , 'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11']
    for joueur in home_team:
        joueur_id = match[joueur]
        if pd.notnull(joueur_id):  # Vérifier si la valeur n'est pas NaN
            joueur_stat = joueur_opti.get_group(joueur_id)
            joueur_stat = joueur_stat[joueur_stat['date'] < date_match]
            joueur_stat = joueur_stat.sort_values(by='date', ascending=False).head(1)
            if joueur_stat.empty:
                overall_rating = 60  # Valeur par défaut si aucune correspondance n'est trouvée
            else:
                overall_rating = joueur_stat['overall_rating'].iloc[0]
        else:
            overall_rating = 60  # Valeur par défaut si la valeur est NaN
        somme = somme + overall_rating
    moy = somme / 11
    return moy


def niv_ext(match):
    date_match = match['date']
    somme = 0
    joueur_stat = pd.DataFrame()
    match_id = match.match_api_id
    away_team = ['away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6'
        , 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11']
    for joueur in away_team:
        joueur_id = match[joueur]
        if pd.notnull(joueur_id):  # Vérifier si la valeur n'est pas NaN
            joueur_stat = joueur_opti.get_group(joueur_id)
            joueur_stat = joueur_stat[joueur_stat['date'] < date_match]
            joueur_stat = joueur_stat.sort_values(by='date', ascending=False).head(1)
            if joueur_stat.empty:
                overall_rating = 60  # Valeur par défaut si aucune correspondance n'est trouvée
            else:
                overall_rating = joueur_stat['overall_rating'].iloc[0]
        else:
            overall_rating = 60# Valeur par défaut si la valeur est NaN
        somme = somme + overall_rating
    moy = somme / 11
    return moy


# je m'interresse a la difference de niveau car finalement c'est cela qui risque d'impacter le plus le résultat
def diff_niv(match):
    niveau_dom = niv_dom(match)
    niveau_ext = niv_ext(match)
    difference = niveau_dom - niveau_ext
    return niveau_dom, niveau_ext, difference


def opti_niv(row):
    niveau_dom, niveau_ext, diff_niveau = diff_niv(row)
    return niveau_dom, niveau_ext, diff_niveau


# fonction qui va dire le résultat, sauf que je vais dire V, D ,N
def resultat_lettre(match):
    but_dom = match.home_team_goal
    but_ext = match.away_team_goal
    # print(but_dom)
    # print(but_ext)
    if (but_dom == but_ext):
        return 1
    if (but_dom > but_ext):
        return 3
    if (but_dom < but_ext):
        return 0
match['resultat'] = match.apply(lambda row: resultat_lettre(row), axis=1)
"""
match['resultat_nom'] = match['resultat'].replace({0: 'Extérieur', 1: 'Match nul', 3: 'Domicile'})

distribution_resultats = match['resultat_nom'].value_counts()

plt.bar(distribution_resultats.index, distribution_resultats.values)
plt.xlabel('Résultat')
plt.ylabel('Nombre de matchs')
plt.title('Distribution des résultats')
plt.show()"""
# pour calculer le nombre de but d'une equipe je vais cree une fonction qui va correctement renvoyer le nombre de but
def but_marque(match, id_team):
    if (match.home_team_api_id == id_team):
        return match.home_team_goal
    else:
        return match.away_team_goal


# meme principe mais avec les but encaissé
def but_encaisse(match, id_team):
    if (match.home_team_api_id == id_team):
        return match.away_team_goal
    else:
        return match.home_team_goal


# ceci va renvoyer 4 nombre, la moyenne du nombre de but marque,celle encaissé des deux équipes
def nombre_but_match(match, match_id):
    date_match = match_id['date']
    equipe_dom_id = match_id.home_team_api_id
    equipe_ext_id = match_id.away_team_api_id
    season_match = match_id.season
    # print(match_id.stage)
    total_match_saison_dom = match[
        ((match.home_team_api_id == equipe_dom_id) | (match.away_team_api_id == equipe_dom_id))
        & (match.season == season_match)]
    total_match_saison_dom = total_match_saison_dom[total_match_saison_dom['date'] < date_match]
    total_match_saison_dom['but_M'] = total_match_saison_dom.apply(lambda row: but_marque(row, equipe_dom_id), axis=1)
    somme_but_dom = total_match_saison_dom['but_M'].sum()
    total_match_saison_dom['but_E'] = total_match_saison_dom.apply(lambda row: but_encaisse(row, equipe_dom_id), axis=1)
    somme_but_enc_dom = total_match_saison_dom['but_E'].sum()
    nombre_matchs_dom = len(total_match_saison_dom)
    # print(somme_but_dom)
    # print(somme_but_enc_dom)
    # print(total_match_saison_dom)
    if (nombre_matchs_dom != 0):
        somme_but_dom = somme_but_dom / nombre_matchs_dom
        somme_but_enc_dom = somme_but_enc_dom / nombre_matchs_dom
    # print(somme_but_dom)
    # print(somme_but_enc_dom)
    total_match_saison_ext = match[
        ((match.home_team_api_id == equipe_ext_id) | (match.away_team_api_id == equipe_ext_id))
        & (match.season == season_match)]
    total_match_saison_ext = total_match_saison_ext[total_match_saison_ext['date'] < date_match]

    total_match_saison_ext['but_M'] = total_match_saison_ext.apply(lambda row: but_marque(row, equipe_ext_id), axis=1)
    somme_but_ext = total_match_saison_ext['but_M'].sum()
    total_match_saison_ext['but_E'] = total_match_saison_ext.apply(lambda row: but_encaisse(row, equipe_ext_id), axis=1)
    somme_but_enc_ext = total_match_saison_ext['but_E'].sum()
    nombre_matchs_ext = len(total_match_saison_ext)
    if (nombre_matchs_ext != 0):
        somme_but_ext = somme_but_ext / nombre_matchs_ext
        somme_but_enc_ext = somme_but_enc_ext / nombre_matchs_ext
    # print(somme_but_ext)
    # print(somme_but_enc_ext)
    if (nombre_matchs_dom == 0 | nombre_matchs_ext == 0):
        return 0, 0, 0, 0, 0
    difference = somme_but_dom + somme_but_enc_ext - somme_but_ext - somme_but_enc_dom
    return somme_but_dom, somme_but_enc_dom, somme_but_ext, somme_but_enc_ext, difference


nombre_but_match(match, match.iloc[25])


def afficher_nombre_match_par_saison(df):
    saison_counts = df['season'].value_counts().sort_index()

    for saison, count in saison_counts.items():
        print(f"Saison {saison}: {count} match(s)")


# Utilisation de la fonction
afficher_nombre_match_par_saison(match)


def point_saison(match, match_id):
    date_match = match_id['date']
    equipe_dom_id = match_id.home_team_api_id
    equipe_ext_id = match_id.away_team_api_id
    season_match = match_id.season
    # on prends les amtch de la saison
    total_match_saison_dom = match[
        ((match.home_team_api_id == equipe_dom_id) | (match.away_team_api_id == equipe_dom_id))
        & (match.season == season_match) & (match.date < date_match)]

    # nb de points de l'equipe à dom
    points_dom = sum(total_match_saison_dom.apply(lambda x: resultat(x, equipe_dom_id), axis=1))
    nombre_matchs_dom = len(total_match_saison_dom)
    if (nombre_matchs_dom == 0):
        return 0, 0, 0
    else:
        moyenne_points_dom = points_dom / nombre_matchs_dom
    # meme chose on prends les match de la saison avant la date
    total_match_saison_ext = match[
        ((match.home_team_api_id == equipe_ext_id) | (match.away_team_api_id == equipe_ext_id))
        & (match.season == season_match) & (match.date < date_match)]

    # nb points equipe à ext
    points_ext = sum(total_match_saison_ext.apply(lambda x: resultat(x, equipe_ext_id), axis=1))
    nombre_matchs_ext = len(total_match_saison_ext)
    if (nombre_matchs_ext == 0):
        return 0, 0, 0
    else:
        moyenne_points_ext = points_ext / nombre_matchs_ext
        diff_point = moyenne_points_dom - moyenne_points_ext
    return diff_point, moyenne_points_dom, moyenne_points_ext


def opti_point(row):
    diff_point, moyenne_points_dom, moyenne_points_ext = point_saison(match, row)
    return diff_point, moyenne_points_dom, moyenne_points_ext


# PERMET DOPTIMISER AU LIEU DE FAIRE APPLIQUER LA FONCTION 3 FOIS
def opti_forme(row):
    forme_diff, forme_dom, forme_ext = diff_form(match, row)
    return forme_dom, forme_ext, forme_diff


def opti_but(row):
    moy_but_dom, moy_but_enc_dom, moy_but_ext, moy_but_enc_ext, difference = nombre_but_match(match, row)
    return moy_but_dom, moy_but_enc_dom, moy_but_ext, moy_but_enc_ext, difference


def latest_team_attributes(row, team_attribue):
    home_team_id = row['home_team_api_id']
    away_team_id = row['away_team_api_id']
    match_date = row['date']
    a = 1
    b = 1
    c = 1
    # Sélection des attributs de l'équipe à domicile avec une date antérieure ou égale à celle du match
    home_attributes = team_attribue[
        (team_attribue['team_api_id'] == home_team_id) & (team_attribue['date'] <= match_date)].tail(1)

    # Sélection des attributs de l'équipe à l'extérieur avec une date antérieure ou égale à celle du match
    away_attributes = team_attribue[
        (team_attribue['team_api_id'] == away_team_id) & (team_attribue['date'] <= match_date)].tail(1)
    # Création des nouvelles colonnes dans le dataframe 'match'
    if (home_attributes['buildUpPlaySpeed'].empty):
        buildUpPlaySpeed_home = 50
        a = 0
    else:
        buildUpPlaySpeed_home = home_attributes['buildUpPlaySpeed'].values[0]
    if (home_attributes['buildUpPlayPassing'].empty):
        buildUpPlayPassing_home = 50
        b = 0
    else:
        buildUpPlayPassing_home = home_attributes['buildUpPlayPassing'].values[0]
    if home_attributes['chanceCreationPassing'].empty:
        chanceCreationPassing_home = 50
    else:
        chanceCreationPassing_home = home_attributes['chanceCreationPassing'].values[0]
    if home_attributes['chanceCreationCrossing'].empty:
        chanceCreationCrossing_home = 50
    else:
        chanceCreationCrossing_home = home_attributes['chanceCreationCrossing'].values[0]
    if home_attributes['chanceCreationShooting'].empty:
        chanceCreationShooting_home = 50
        c = 0
    else:
        chanceCreationShooting_home = home_attributes['chanceCreationShooting'].values[0]

    if home_attributes['defencePressure'].empty:
        defencePressure_home = 50
    else:
        defencePressure_home = home_attributes['defencePressure'].values[0]

    if home_attributes['defenceAggression'].empty:
        defenceAggression_home = 50
    else:
        defenceAggression_home = home_attributes['defenceAggression'].values[0]
    if away_attributes['buildUpPlaySpeed'].empty:
        buildUpPlaySpeed_away = 50
        a = 0
    else:
        buildUpPlaySpeed_away = away_attributes['buildUpPlaySpeed'].values[0]
    if away_attributes['buildUpPlayPassing'].empty:
        buildUpPlayPassing_away = 50
        b = 0
    else:
        buildUpPlayPassing_away = away_attributes['buildUpPlayPassing'].values[0]

    if away_attributes['chanceCreationPassing'].empty:
        chanceCreationPassing_away = 50
    else:
        chanceCreationPassing_away = away_attributes['chanceCreationPassing'].values[0]
    if away_attributes['chanceCreationCrossing'].empty:
        chanceCreationCrossing_away = 50
    else:
        chanceCreationCrossing_away = away_attributes['chanceCreationCrossing'].values[0]
    if away_attributes['chanceCreationShooting'].empty:
        chanceCreationShooting_away = 50
        c = 0
    else:
        chanceCreationShooting_away = away_attributes['chanceCreationShooting'].values[0]
    if away_attributes['defencePressure'].empty:
        defencePressure_away = 50
    else:
        defencePressure_away = away_attributes['defencePressure'].values[0]
    if away_attributes['defenceAggression'].empty:
        defenceAggression_away = 50
    else:
        defenceAggression_away = away_attributes['defenceAggression'].values[0]
    if (a == 0):
        buildUpPlaySpeed = 0
    else:
        buildUpPlaySpeed = buildUpPlaySpeed_home - buildUpPlaySpeed_away
    if (b == 0):
        buildUpPlayPassing = 0
    else:
        buildUpPlayPassing = buildUpPlayPassing_home - buildUpPlayPassing_away

    chanceCreationPassing = chanceCreationPassing_home - chanceCreationPassing_home
    chanceCreationCrossing = chanceCreationCrossing_home - chanceCreationCrossing_away
    if (c == 0):
        chanceCreationShooting = 0
    else:
        chanceCreationShooting = chanceCreationShooting_home - chanceCreationShooting_away
    defencePressure = defencePressure_home - defencePressure_away
    defenceAggression = defenceAggression_home - defenceAggression_away
    return buildUpPlaySpeed, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting \
        , defencePressure, defenceAggression


def opti_team_attributes(row):
    buildUpPlaySpeed, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting \
        , defencePressure, defenceAggression = latest_team_attributes(row, team_attribue)
    return buildUpPlaySpeed, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting \
        , defencePressure, defenceAggression


def pourcentage_matchs_nuls(match, match_id):
    date_match = match_id['date']
    equipe_dom_id = match_id.home_team_api_id
    equipe_ext_id = match_id.away_team_api_id
    season_match = match_id.season

    # On récupère les matchs de la saison pour l'équipe à domicile
    total_match_saison_dom = match[
        ((match.home_team_api_id == equipe_dom_id) | (match.away_team_api_id == equipe_dom_id))
        & (match.season == season_match) & (match.date < date_match)]

    # On compte le nombre de matchs nuls pour l'équipe à domicile
    matchs_nuls_dom = sum(total_match_saison_dom['resultat'] == 1)
    nombre_matchs_dom = len(total_match_saison_dom)
    pourcentage_nuls_dom = matchs_nuls_dom / nombre_matchs_dom if nombre_matchs_dom > 0 else 0

    # On récupère les matchs de la saison pour l'équipe à l'extérieur
    total_match_saison_ext = match[
        ((match.home_team_api_id == equipe_ext_id) | (match.away_team_api_id == equipe_ext_id))
        & (match.season == season_match) & (match.date < date_match)]
    # On compte le nombre de matchs nuls pour l'équipe à l'extérieur
    matchs_nuls_ext = sum(total_match_saison_ext['resultat'] == 1)
    nombre_matchs_ext = len(total_match_saison_ext)
    pourcentage_nuls_ext = matchs_nuls_ext / nombre_matchs_ext if nombre_matchs_ext > 0 else 0

    return pourcentage_nuls_dom, pourcentage_nuls_ext,matchs_nuls_dom,matchs_nuls_ext
match[['nul_dom','nul_ext','len_nul_dom','len_nul_ext']]=match.apply(lambda row :pourcentage_matchs_nuls(match,row), axis=1, result_type='expand')
print(match.iloc[640])
match[['buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing', 'chanceCreationCrossing',
       'chanceCreationShooting' \
    , 'defencePressure', 'defenceAggression']] = match.apply(lambda row: opti_team_attributes(row), axis=1,
                                                             result_type='expand')

start_time = time.time()
match[['forme_dom', 'forme_ext', 'forme_diff']] = match.apply(lambda row: opti_forme(row), axis=1, result_type='expand')

print("Les features 'forme_dom', 'forme_ext' et 'forme_diff' ont été calculées en %s secondes" % (
            time.time() - start_time))

start_time = time.time()
match[['niv_dom', 'niv_ext', 'diff_niv']] = match.apply(lambda row: opti_niv(row), axis=1, result_type='expand')
print("Les features 'niv_dom','niv_ext','diff_niv' a été calculée en %s secondes" % (time.time() - start_time))

start_time = time.time()
match[['diff_point', 'moyenne_points_dom', 'moyenne_points_ext']] = match.apply(lambda row: opti_point(row), axis=1,
                                                                                result_type='expand')

print("Les features 'diff_point', 'moyenne_points_dom' et 'moyenne_points_ext' ont été calculées en %s secondes" % (
            time.time() - start_time))
start_time = time.time()
match[['moy_but_dom', 'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference']] = match.apply(
    lambda row: opti_but(row), axis=1, result_type='expand')

print(
    "Les features 'moy_but_dom', 'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext' et 'difference' ont été calculées en %s secondes" % (
                time.time() - start_time))
print(match.iloc[44])


weights = [1, 7, 1, 1, 1, 7, 1, 1, 1, 1, 5, 1, 1, 5,5]
match_ligue1= match[match['country_id'].isin([4769])]
match_bundes= match[match['country_id'].isin([7809])]
match_seria=match[match['country_id'].isin([10257])]
match_liga=match[match['country_id'].isin([21518])]
match_premier=match[match['country_id'].isin([1729])]
leagues = {
    'Ligue 1': match_ligue1,
    'Bundesliga': match_bundes,
    'Liga': match_liga,
    'Serie A': match_seria,
    'Premier League': match_premier
}

weights = [4,4,8,1, 1, 5, 1, 1, 1, 1, 5, 1, 1, 5,3,3,1,1,1]#1, 7, 1,

scaler = StandardScaler()

"""---------------------------------------------------------------------------------------------------------------------

                                                KNN AVEC AFFICHAGE PROPRE

------------------------------------------------------------------------------------------------------------------------
"""
param_grid = {'n_neighbors': range(1, 200)}
accuracies_knn = {'Ligue 1': {'H': [], 'D': [], 'A': []},
                  'Bundesliga': {'H': [], 'D': [], 'A': []},
                  'Liga': {'H': [], 'D': [], 'A': []},
                  'Serie A': {'H': [], 'D': [], 'A': []},
                  'Premier League': {'H': [], 'D': [], 'A': []}}

f1_scores = {'Ligue 1': {'H': [], 'D': [], 'A': []},
             'Bundesliga': {'H': [], 'D': [], 'A': []},
             'Liga': {'H': [], 'D': [], 'A': []},
             'Serie A': {'H': [], 'D': [], 'A': []},
             'Premier League': {'H': [], 'D': [], 'A': []}}
confusion_matrices_knn = []
fig, ax = plt.subplots()
budgets_by_league = {}
"""
for league, data in leagues.items():
    X = data[['niv_dom','niv_ext','diff_niv','forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
              'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
              'moyenne_points_ext', 'diff_point','nul_dom','nul_ext']]

    X_scaled = scaler.fit_transform(X)
    X_scaled = np.hstack((X_scaled, data[['B365H', 'B365D', 'B365A']]))
    X_scaled_w = X_scaled * weights
    y = pd.DataFrame()
    y['resultat'] = data.apply(lambda row: resultat_lettre(row), axis=1)
    # Diviser les données entre les données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_w, y.values.ravel(), test_size=0.25, random_state=42)

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_k = grid_search.best_params_['n_neighbors']
    best_score = grid_search.best_score_

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)

    y_pred = best_knn.predict(X_test)
    y_pred = y_pred.astype(str)
    y_test = y_test.astype(str)
    total_bets = 0
    budget = 100
    correct_predictions = 0

    for prediction, cote, true_label in zip(y_pred, X_test[:, -3:], y_test):
        if prediction == '1':
            cotes = cote[1]  # ('B365D')
        elif prediction == '3':
            cotes = cote[0]  # ('B365H')
        else:
            cotes = cote[2]  # ('B365A')
        bet_amount = 100
        if prediction == true_label:
            total_bets += bet_amount
            potential_winning = bet_amount * (cotes - 1)
            budget += potential_winning
            correct_predictions +=1
        else:
            total_bets += bet_amount
            budget -= bet_amount
    accuracy = correct_predictions / len(y_test)
    budgets_by_league[league] = budget
    print("Ligue:", league)
    print("Exactitude sur l'ensemble de test:", accuracy)
    print("Total des mises:", total_bets)
    print("Budget final:", budget)
    print("------------------")
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_knn[league]['H'].append(accuracy_score(y_test[y_test == '3'], y_pred[y_test == '3']))
    accuracies_knn[league]['D'].append(accuracy_score(y_test[y_test == '1'], y_pred[y_test == '1']))
    accuracies_knn[league]['A'].append(accuracy_score(y_test[y_test == '0'], y_pred[y_test == '0']))
    f1_scores[league]['H'].append(f1_score(y_test, y_pred, average='weighted'))
    f1_scores[league]['D'].append(f1_score(y_test, y_pred, average='weighted'))
    f1_scores[league]['A'].append(f1_score(y_test, y_pred, average='weighted'))
    # Calculer la matrice de confusion
    classes = ['0', '1', '3']
    confusion_mlp = confusion_matrix(y_test, y_pred, labels=classes)
    confusion_matrices_knn.append(confusion_mlp)
    print("Ligue:", league)
    print("Meilleurs paramètres:", grid_search.best_params_)
    print("Exactitude sur l'ensemble de test:", accuracy)
    print("------------------")
    cv_scores = cross_val_score(best_knn, X_scaled, y.values.ravel(), cv=5)
    print("Scores de validation croisée:", cv_scores)
    print("Score moyen de validation croisée:", np.mean(cv_scores))
    print("------------------")
    ax.bar(league, accuracy)
    ax.text(league, accuracy + 0.01, str(round(accuracy, 2)), ha='center', va='bottom')

ax.set_xlabel('Ligue')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Different Leagues with KNN')
plt.show()
# Affichage des précisions et F1-score
labels = ['A', 'D', 'H']
labels2 = ['H', 'D', 'A']
leagues_names = ['Ligue 1', 'Bundesliga', 'Liga', 'Serie A', 'Premier League']
accuracies_list = np.array(
    [accuracies_knn[league]['H'] + accuracies_knn[league]['D'] + accuracies_knn[league]['A'] for league in
     accuracies_knn])
f1_scores_list = np.array(
    [f1_scores[league]['H'] + f1_scores[league]['D'] + f1_scores[league]['A'] for league in f1_scores])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# plot les bbudgets finaux
plt.figure(figsize=(10, 6))

leagues_names = list(budgets_by_league.keys())
budgets = list(budgets_by_league.values())

colors = ['red' if budget < 0 else 'green' for budget in budgets]

bars = plt.bar(leagues_names, budgets, color=colors)
plt.xlabel('Ligue')
plt.ylabel('Budget final')
plt.title('Budget final par ligue')
plt.xticks(rotation=45)
plt.tight_layout()

for bar, budget in zip(bars, budgets):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{budget:.2f}',
             ha='center', va='bottom', color='black')

plt.show()
# précisions par type de résultat
bar_width = 0.2
index = np.arange(len(leagues_names))

for i in range(len(labels2)):
    ax1.bar(index + i * bar_width, accuracies_list[:, i], bar_width, label=labels2[i])
    for j, value in enumerate(accuracies_list[:, i]):
        ax1.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax1.set_xlabel('Championnat')
ax1.set_ylabel('Précision')
ax1.set_title('Précision des prédictions par type de résultat et par championnat')
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(leagues_names)
ax1.legend()

# F1-scores par type de résultat
f1_scores_list = np.array(
    [f1_scores[league]['H'] + f1_scores[league]['D'] + f1_scores[league]['A'] for league in f1_scores])

for i in range(len(labels2)):
    ax2.bar(index + i * bar_width, f1_scores_list[:, i], bar_width, label=labels2[i])
    for j, value in enumerate(f1_scores_list[:, i]):
        ax2.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax2.set_xlabel('Championnat')
ax2.set_ylabel('F1-score')
ax2.set_title('F1-score des prédictions par type de résultat et par championnat')
ax2.set_xticks(index + bar_width)
ax2.set_xticklabels(leagues_names)
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for i in range(len(leagues_names)):
    fig, ax = plt.subplots(figsize=(6, 6))

    confusion_matrix = confusion_matrices_knn[i]

    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')

    for j in range(len(labels)):
        for k in range(len(labels)):
            text = ax.text(k, j, confusion_matrix[j, k], ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vraie étiquette')
    ax.set_title("Matrice de confusion - KNN - " + leagues_names[i])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()"""

"""---------------------------------------------------------------------------------------------------------------------

                                  MLP AVEC AFFICHAGE ET AVEC SMOTE 

------------------------------------------------------------------------------------------------------------------------
"""

budgets_by_league = {}
param_grid_mlp = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'max_iter': [1500],
    'random_state': [42]
}
fig, ax = plt.subplots()
hidden_layer_size = [
    (100,),
    (100, 50),
    (50, 50),
    (100, 100, 50),
    (50, 25),
    (100, 50, 25),
    (200, 100, 50),
    (150, 100, 75, 50),
    (50, 50, 50)
]
accuracies_mlp = {'Ligue 1': {'H': [], 'D': [], 'A': []},
                  'Bundesliga': {'H': [], 'D': [], 'A': []},
                  'Liga': {'H': [], 'D': [], 'A': []},
                  'Serie A': {'H': [], 'D': [], 'A': []},
                  'Premier League': {'H': [], 'D': [], 'A': []}}
f1_scores = {'Ligue 1': {'H': [], 'D': [], 'A': []},
             'Bundesliga': {'H': [], 'D': [], 'A': []},
             'Liga': {'H': [], 'D': [], 'A': []},
             'Serie A': {'H': [], 'D': [], 'A': []},
             'Premier League': {'H': [], 'D': [], 'A': []}}
confusion_matrices_mlp = []

for league, data in leagues.items():
    X = data[['niv_dom', 'niv_ext', 'diff_niv', 'forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
              'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
              'moyenne_points_ext', 'diff_point','nul_dom','nul_ext']]

    X_scaled = scaler.fit_transform(X)
    X_scaled = np.hstack((X_scaled, data[['B365H', 'B365D', 'B365A']]))
    y = pd.DataFrame()
    y['resultat'] = data.apply(lambda row: resultat_lettre(row), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.25, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    grid_search_mlp = GridSearchCV(MLPClassifier(hidden_layer_sizes=(100, 50)), param_grid_mlp, cv=5)
    grid_search_mlp.fit(X_train_smote, y_train_smote)
    best_model = grid_search_mlp.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = y_pred.astype(str)
    y_test = y_test.astype(str)
    """seuil = 0.40
    y_pred_prob = best_model.predict_proba(X_test)
    print(y_pred_prob)
    y_pred = []
    for proba in y_pred_prob:
        if max(proba) > seuil:
            if np.argmax(proba) == 2:
                y_pred.append('3')
            else:
                y_pred.append(str(np.argmax(proba)))
        else:
            y_pred.append('1')
    y_test = y_test.astype(str)
    y_pred = np.array(y_pred)
    y_pred = y_pred.astype(str)"""
    total_bets = 0
    budget = 100
    correct_predictions =0
    for prediction, cote, true_label in zip(y_pred, X_test[:, -3:], y_test):
        if prediction == '1':
            cotes = cote[1]  # ('B365D')
        elif prediction == '3':
            cotes = cote[0]  # ('B365H')
        else:
            cotes = cote[2]  # ('B365A')
        bet_amount = 100
        if prediction == true_label:
            total_bets += bet_amount
            potential_winning = bet_amount * (cotes-1)
            budget += potential_winning
            correct_predictions +=1
        else:
            total_bets += bet_amount
            budget -= bet_amount
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = correct_predictions / len(y_test)
    budgets_by_league[league] = budget
    print("Ligue:", league)
    print("Exactitude sur l'ensemble de test:", accuracy)
    print("Total des mises:", total_bets)
    print("Budget final:", budget)
    print("------------------")
    accuracies_mlp[league]['H'].append(accuracy_score(y_test[y_test == '3'], y_pred[y_test == '3']))
    accuracies_mlp[league]['D'].append(accuracy_score(y_test[y_test == '1'], y_pred[y_test == '1']))
    accuracies_mlp[league]['A'].append(accuracy_score(y_test[y_test == '0'], y_pred[y_test == '0']))
    f1_scores[league]['H'].append(f1_score(y_test, y_pred, average='weighted'))
    f1_scores[league]['D'].append(f1_score(y_test, y_pred, average='weighted'))
    f1_scores[league]['A'].append(f1_score(y_test, y_pred, average='weighted'))
    # Calculer la matrice de confusion
    classes = ['0', '1', '3']
    confusion_mlp = confusion_matrix(y_test, y_pred, labels=classes)
    confusion_matrices_mlp.append(confusion_mlp)
    print("Ligue:", league)
    print("Meilleurs paramètres:", grid_search_mlp.best_params_)
    print("Exactitude sur l'ensemble de test:", accuracy)
    print("------------------")
    cv_scores = cross_val_score(best_model, X_scaled, y.values.ravel(), cv=5)
    print("Scores de validation croisée:", cv_scores)
    print("Score moyen de validation croisée:", np.mean(cv_scores))
    print("------------------")
    ax.bar(league, accuracy)
    ax.text(league, accuracy + 0.01, str(round(accuracy, 2)), ha='center', va='bottom')
plt.figure(figsize=(10, 6))

leagues_names = list(budgets_by_league.keys())
budgets = list(budgets_by_league.values())

colors = ['red' if budget < 0 else 'green' for budget in budgets]

bars = plt.bar(leagues_names, budgets, color=colors)
plt.xlabel('Ligue')
plt.ylabel('benefice/perte')
plt.title('benefice / perte par ligue ( MLP)')
plt.xticks(rotation=45)
plt.tight_layout()

for bar, budget in zip(bars, budgets):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{budget:.2f}',
             ha='center', va='bottom', color='black')

plt.show()

ax.set_xlabel('Ligue')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Different Leagues with MLP')
plt.show()
# Affichage des précisions et F1-score
labels = ['A', 'D', 'H']
labels2=['H','D','A']
leagues_names = ['Ligue 1', 'Bundesliga', 'Liga', 'Serie A', 'Premier League']
accuracies_list = np.array([accuracies_mlp[league]['H'] + accuracies_mlp[league]['D'] + accuracies_mlp[league]['A'] for league in accuracies_mlp])
f1_scores_list = np.array([f1_scores[league]['H'] + f1_scores[league]['D'] + f1_scores[league]['A'] for league in f1_scores])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# précisions par type de résultat
bar_width = 0.2
index = np.arange(len(leagues_names))

for i in range(len(labels2)):
    ax1.bar(index + i * bar_width, accuracies_list[:, i], bar_width, label=labels2[i])
    for j, value in enumerate(accuracies_list[:, i]):
        ax1.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax1.set_xlabel('Championnat')
ax1.set_ylabel('Précision')
ax1.set_title('Précision des prédictions par type de résultat et par championnat')
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(leagues_names)
ax1.legend()

# F1-scores par type de résultat
f1_scores_list = np.array([f1_scores[league]['H'] + f1_scores[league]['D'] + f1_scores[league]['A'] for league in f1_scores])

for i in range(len(labels2)):
    ax2.bar(index + i * bar_width, f1_scores_list[:, i], bar_width, label=labels2[i])
    for j, value in enumerate(f1_scores_list[:, i]):
        ax2.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax2.set_xlabel('Championnat')
ax2.set_ylabel('F1-score')
ax2.set_title('F1-score des prédictions par type de résultat et par championnat')
ax2.set_xticks(index + bar_width)
ax2.set_xticklabels(leagues_names)
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for i in range(len(leagues_names)):
    fig, ax = plt.subplots(figsize=(6, 6))

    confusion_matrix = confusion_matrices_mlp[i]

    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')

    for j in range(len(labels)):
        for k in range(len(labels)):
            text = ax.text(k, j, confusion_matrix[j, k], ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vraie étiquette')
    ax.set_title("Matrice de confusion - MLP - " + leagues_names[i])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

"""--------------------------------------------------------------------------------------------------------------------
 
                                  RANDOM FOREST AVEC AFFICHAGE PROPRE

---------------------------------------------------------------------------------------------------------------------"""

fig, ax = plt.subplots()

"""
best_params_ligue1 = {'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 300}
best_params_bundesliga = {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 350}
best_params_liga = {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
best_params_serieA = {'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}
best_params_premierLeague = {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 250}
"""

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2]
}

accuracies = {'Ligue 1': {'H': [], 'D': [], 'A': []},
              'Bundesliga': {'H': [], 'D': [], 'A': []},
              'Liga': {'H': [], 'D': [], 'A': []},
              'Serie A': {'H': [], 'D': [], 'A': []},
              'Premier League': {'H': [], 'D': [], 'A': []}}
confusion_matrices = []
f1_scores = []
class_counts = {'0': 0, '1': 0, '3': 0}
budgets_by_league = {}
"""
for league, data in leagues.items():
    X = data[['niv_dom', 'niv_ext', 'diff_niv', 'forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
              'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
              'moyenne_points_ext', 'diff_point','stage','nul_dom','nul_ext']]

    X_scaled = scaler.fit_transform(X)
    X_scaled = np.hstack((X_scaled, data[['B365H', 'B365D', 'B365A']]))
    y = pd.DataFrame()
    y['resultat'] = data.apply(lambda row: resultat_lettre(row), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.25, random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_params = grid_search.best_params_
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    y_pred=model.predict(X_test)
    y_pred = y_pred.astype(str)
    y_test = y_test.astype(str)
    total_bets = 0
    budget = 100
    correct_predictions=0
    for prediction, cote, true_label in zip(y_pred, X_test[:, -3:], y_test):
        if prediction == '1':
            cotes = cote[1]  # ('B365D')
        elif prediction == '3':
            cotes = cote[0]  # ('B365H')
        else:
            cotes = cote[2]  # ('B365A')
        bet_amount = 100
        if prediction == true_label:
            total_bets += bet_amount
            potential_winning = bet_amount * (cotes-1)
            budget += potential_winning
            correct_predictions += 1
        else:
            total_bets += bet_amount
            budget -= bet_amount
    accuracy = correct_predictions / len(y_test)
    budgets_by_league[league] = budget
    print("Ligue:", league)
    print("Exactitude sur l'ensemble de test:", accuracy)
    print("Total des mises:", total_bets)
    print("Budget final:", budget)
    print("------------------")
    y_pred_prob = model.predict_proba(X_test)
    seuil = 0.42
    y_pred = []
    for proba in y_pred_prob:
        if np.argmax(proba) == 0:
            y_pred.append('0')
        elif max(proba) > seuil:
            if np.argmax(proba) == 2:
                y_pred.append('3')
            else:
                y_pred.append(str(np.argmax(proba)))
        else:
            y_pred.append('1')
    test_score = model.score(X_test, y_test)
    print("Ligue:", league)
    print("Meilleurs paramètres:", best_params)
    print("Exactitude sur l'ensemble de test:", test_score)
    print("------------------")
    y_pred = np.array(y_pred)
    y_pred = y_pred.astype(str)
    y_test = y_test.astype(str)
    # Calcul des précisions par résultat
    accuracies[league]['H'].append(accuracy_score(y_test[y_test == '3'], y_pred[y_test == '3']))
    accuracies[league]['D'].append(accuracy_score(y_test[y_test == '1'], y_pred[y_test == '1']))
    accuracies[league]['A'].append(accuracy_score(y_test[y_test == '0'], y_pred[y_test == '0']))
    print(accuracies[league])
    print(y_pred)
    print("y test : %s",y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(f1)

    # Définir les classes possibles
    classes = ['0', '1', '3']

    # matrice confusion
    confusion = confusion_matrix(y_test, y_pred, labels=classes)

    print("Matrice de confusion:")
    print(confusion)
    print("------------------")
    confusion_matrices.append(confusion)
    indices = np.argsort(feature_importance)[::-1]
    features = X.columns[indices]
    importance = feature_importance[indices]

    # Tracez les caractéristiques les plus importantes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), features, rotation=90)
    plt.xlabel('Caractéristiques')
    plt.ylabel('Importance')
    plt.title('Importance des caractéristiques : Random Forest - {}'.format(league))
    plt.show()""""""
plt.figure(figsize=(10, 6))

leagues_names = list(budgets_by_league.keys())
budgets = list(budgets_by_league.values())

colors = ['red' if budget < 0 else 'green' for budget in budgets]

bars = plt.bar(leagues_names, budgets, color=colors)
plt.xlabel('Ligue')
plt.ylabel('benefice/perte')
plt.title('benefice / perte par ligue ( RF)')
plt.xticks(rotation=45)
plt.tight_layout()

for bar, budget in zip(bars, budgets):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{budget:.2f}',
             ha='center', va='bottom', color='black')

plt.show()
# Préparation des données pour l'affichage
labels = ['A', 'D', 'H']
labels2 = ['H', 'D', 'A']
leagues_names = ['Ligue 1', 'Bundesliga', 'Liga', 'Serie A', 'Premier League']
accuracies_list = np.array([accuracies[league]['H'] + accuracies[league]['D'] + accuracies[league]['A'] for league in accuracies])
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Classe')
plt.ylabel('Nombre d\'échantillons')
plt.title('Nombre d\'échantillons par classe pour toutes les ligues')
plt.show()
# Affichage des précisions par type de résultat pour chaque championnat
fig, ax = plt.subplots()
bar_width = 0.2
index = np.arange(len(leagues_names))

for i in range(len(labels2)):
    ax.bar(index + i * bar_width, accuracies_list[:, i], bar_width, label=labels2[i])
    for j, value in enumerate(accuracies_list[:, i]):
        ax.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax.set_xlabel('Championnat')
ax.set_ylabel('Précision')
ax.set_title('Précision des prédictions par type de résultat et par championnat')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(leagues_names)
ax.legend()

plt.xticks(rotation=45)
plt.show()
# Affichage du F1-score pour chaque ligue
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(leagues)), f1_scores)

plt.xticks(np.arange(len(leagues)), leagues.keys())
plt.xlabel('Ligue')
plt.ylabel('F1 Score')
plt.title('F1 Score pour chaque ligue')
plt.show()
# Affichage des matrices de confusion
for i in range(len(leagues_names)):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Sélectionner la matrice de confusion correspondante à un championnat spécifique
    confusion_matrice = confusion_matrices[i]

    # Afficher la matrice de confusion
    im = ax.imshow(confusion_matrice, interpolation='nearest', cmap='Blues')

    # Afficher les valeurs dans les cases
    for j in range(len(labels)):
        for k in range(len(labels)):
            text = ax.text(k, j, confusion_matrice[j, k], ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vraie étiquette')
    ax.set_title("Matrice de confusion - RF - " + leagues_names[i])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

"""
""""""
"""--------------------------PLOT LES ACCURACY PAR CHAMP ---------------------------------------

    ax.bar(league, test_score)
    ax.text(league, test_score + 0.01, str(round(test_score, 2)), ha='center', va='bottom')"""
""""""    """---------------------------PLOT ACCURACY PAR JOURNEE ----------------------------------------
    y_pred_with_resultat_journee = pd.DataFrame({'resultat_pred': y_pred, 'resultat': y_test, 'journée': X_test[:, -1]})

    # Calculer l'exactitude par journée
    accuracy_by_journee = y_pred_with_resultat_journee.groupby('journée').apply(lambda x: accuracy_score(x['resultat'], x['resultat_pred']))

    # Afficher la courbe d'accuracy par journée
    plt.plot(accuracy_by_journee.index, accuracy_by_journee.values, label=league)"""

ax.set_xlabel('Ligue')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Different Leagues with Random Forest')
""" ----------------------------------------------calcul accuracy par journée par championnat*------------
plt.xlabel('Journée')
plt.ylabel('Accuracy')
plt.title('Accuracy par journée pour chaque championnat')
plt.legend()
plt.show()"""
"""

#on va choisir les colonnes les plus importantes pour le pca
#on tranforme en PCA la data pour avoir que deux params
pca=PCA(n_components=2)
x_pca=pca.fit_transform(X_scaled)#entraine model pca avec données, transforme ensuite x en x_pca
plt.figure()
plt.scatter(x_pca[:,0],x_pca[:,1],c=y['resultat'])#les deux colonnes car pca les as réduits en deux colonnes
plt.show()"""
"""----------------------------------------------------------------------------------------------------------------
ON A FINI DE COMPLETER NOS DONNEES L'OBJECTIF MAINTENANT CEST DE FAIRE NOS CLASSIFIERS GRACE AU NOUVELLES VARIABLES OBTENUES

-----------------------------------------------------------------------------------------------------------------"""
# -------------------------------------------------------------------------------------------------------------------
"""
# diviser les donnée entre les données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled_ligue1, y.values.ravel(), test_size=0.25, random_state=42)
# valeurs de k à tester
param_grid = {'n_neighbors': range(1, 200)}

knn = KNeighborsClassifier()

# on ba essayer de trouver le meilleur k par cross validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# on trouve le meilleur k
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

# on utilise le k trouver precedement
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

test_score = best_knn.score(X_test, y_test)

print("k optimal :", best_k)
print("Score de validation croisée:", best_score)
print("Score sur l'ensemble de test:", test_score)

param_grid = {'n_estimators': [250, 300, 350, 400, 450, 500]}

rf = RandomForestClassifier()

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# pour trouver le n_estimators adequats
best_estimators = grid_search.best_params_['n_estimators']
best_score = grid_search.best_score_
# on fait la demarche avec nos valeur
best_rf = RandomForestClassifier(n_estimators=best_estimators)
best_rf.fit(X_train, y_train)

test_score = best_rf.score(X_test, y_test)

print("Meilleur nombre d'estimateurs trouvé:", best_estimators)
print("Score de validation croisée:", best_score)
print("Score sur l'ensemble de test:", test_score)
"""
"""
scaler = StandardScaler()
X = match[['niv_dom','niv_ext','diff_niv','forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
              'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
              'moyenne_points_ext', 'diff_point']]
X_scaled = scaler.fit_transform(X)
y = pd.DataFrame()
y['resultat'] = match.apply(lambda row: resultat_lettre(row), axis=1)
# Diviser les données entre les données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.25, random_state=42)
activations = ['identity', 'logistic', 'tanh', 'relu']
solvers = ['lbfgs', 'sgd', 'adam']

accuracy_scores = np.zeros((len(activations), len(solvers)))

for i, activation in enumerate(activations):
    for j, solver in enumerate(solvers):
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation=activation, solver=solver, max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores[i, j] = accuracy

# Affichage du graphique
fig, ax = plt.subplots()
im = ax.imshow(accuracy_scores, cmap='viridis')

# Ajout des ticks et des labels
ax.set_xticks(np.arange(len(solvers)))
ax.set_yticks(np.arange(len(activations)))
ax.set_xticklabels(solvers)
ax.set_yticklabels(activations)
plt.xlabel('Solver')
plt.ylabel('Activation')

# Affichage des valeurs sur les barres
for i in range(len(activations)):
    for j in range(len(solvers)):
        text = ax.text(j, i, round(accuracy_scores[i, j], 2),
                       ha="center", va="center", color="w")

# Titre du graphique
plt.title('Accuracy scores for different activations and solvers')

# Affichage de la barre de couleur
plt.colorbar(im)

# Affichage du graphique
plt.show()
""""""
hidden_layer_sizes = [
    (100,),
    (100, 50),
    (50, 50),
    (100, 100, 50),
    (50, 25),
]
mlp = MLPClassifier(activation='identity', solver='sgd', max_iter=1000, random_state=42)

# Paramètres de la recherche grid

param_grid = {'hidden_layer_sizes': hidden_layer_sizes}

# Création de l'objet GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, cv=5)

# Exécution de la recherche grid sur les données d'entraînement
grid_search.fit(X_train, y_train)

# Meilleur modèle trouvé
best_model = grid_search.best_estimator_

# Prédictions sur l'ensemble de test avec le meilleur modèle
y_pred = best_model.predict(X_test)

# Évaluation de la performance du meilleur modèle
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude sur l'ensemble de test:", accuracy)

# Meilleurs paramètres trouvés
print("Meilleurs paramètres:", grid_search.best_params_)"""
""""# Créer une figure pour afficher les résultats
fig, ax = plt.subplots()

# Parcourir chaque ligue
for league, data in leagues.items():
    X = data[['niv_dom', 'niv_ext', 'diff_niv', 'forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
              'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
              'moyenne_points_ext', 'diff_point']]

    y = pd.DataFrame()
    y['resultat'] = data.apply(lambda row: resultat_lettre(row), axis=1)
    # Standardiser les caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled * weights

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.25, random_state=42)

    # Créer le modèle GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Prédire les résultats sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer l'exactitude
    accuracy = accuracy_score(y_test, y_pred)

    # Afficher les résultats
    print("Ligue:", league)
    print("Exactitude sur l'ensemble de test:", accuracy)
    print("------------------")

    # Ajouter la barre et le texte correspondant à l'exactitude sur la figure
    ax.bar(league, accuracy)
    ax.text(league, accuracy + 0.01, str(round(accuracy, 2)), ha='center', va='bottom')

# Définir les étiquettes et le titre de la figure
ax.set_xlabel('Ligue')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Different Leagues with GaussianNB')

# Afficher la figure
plt.show()
"""

accuracies = {'Ligue 1': {'H': [], 'D': [], 'A': []},
              'Bundesliga': {'H': [], 'D': [], 'A': []},
              'Liga': {'H': [], 'D': [], 'A': []},
              'Serie A': {'H': [], 'D': [], 'A': []},
              'Premier League': {'H': [], 'D': [], 'A': []}}

confusion_matrices_svm = []

param_grid = {
    'C': [0.1, 1, 10,3],
    'kernel': ['linear', 'rbf','polynomial'],
    'gamma': [0.1, 0.01, 0.001]
}
budgets_by_league = {}
for league, data in leagues.items():
    X = data[['niv_dom', 'niv_ext', 'diff_niv', 'forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
              'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
              'moyenne_points_ext', 'diff_point', 'stage','nul_dom','nul_ext']]

    X_scaled = scaler.fit_transform(X)
    X_scaled = np.hstack((X_scaled, data[['B365H', 'B365D', 'B365A']]))
    y = pd.DataFrame()
    y['resultat'] = data.apply(lambda row: resultat_lettre(row), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.25, random_state=42)

    svm_model = SVC()

    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    model = SVC(**best_params)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(str)
    y_test = y_test.astype(str)
    total_bets = 0
    budget = 100
    correct_predictions=0
    for prediction, cote, true_label in zip(y_pred, X_test[:, -3:], y_test):
        if prediction == '1':
            cotes = cote[1]  # ('B365D')
        elif prediction == '3':
            cotes = cote[0]  # ('B365H')
        else:
            cotes = cote[2]  # ('B365A')
        bet_amount = 100
        if prediction == true_label:
            total_bets += bet_amount
            potential_winning = bet_amount * (cotes-1)
            budget += potential_winning
            correct_predictions+=1
        else:
            total_bets += bet_amount
            budget -= bet_amount
    accuracy=correct_predictions/len(y_test)
    budgets_by_league[league] = budget
    print("Ligue:", league)
    print("Exactitude sur l'ensemble de test:", accuracy)
    print("Total des mises:", total_bets)
    print("Budget final:", budget)
    print("------------------")
    # on stock les différente accuracy afin de voir nos prédiction et de les affiché
    accuracies[league]['H'].append(accuracy_score(y_test[y_test == '3'], y_pred[y_test == '3']))
    accuracies[league]['D'].append(accuracy_score(y_test[y_test == '1'], y_pred[y_test == '1']))
    accuracies[league]['A'].append(accuracy_score(y_test[y_test == '0'], y_pred[y_test == '0']))

    # Calcul la matrice de confusion
    classes = ['0', '1', '3']
    confusion_svm = confusion_matrix(y_test, y_pred, labels=classes)
    confusion_matrices_svm.append(confusion_svm)
    print("Ligue:", league)
    print("Meilleurs paramètres:", best_params)
    print("Exactitude sur l'ensemble de test:", test_score)
    print("------------------")

plt.figure(figsize=(10, 6))

leagues_names = list(budgets_by_league.keys())
budgets = list(budgets_by_league.values())

colors = ['red' if budget < 0 else 'green' for budget in budgets]

bars = plt.bar(leagues_names, budgets, color=colors)
plt.xlabel('Ligue')
plt.ylabel('benefice/perte')
plt.title('benefice / perte par ligue ( SVM)')
plt.xticks(rotation=45)
plt.tight_layout()

for bar, budget in zip(bars, budgets):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{budget:.2f}',
             ha='center', va='bottom', color='black')

plt.show()
# Affichage des précisions
labels = ['A', 'D', 'H']
leagues_names = ['Ligue 1', 'Bundesliga', 'Liga', 'Serie A', 'Premier League']
accuracies_list = np.array([accuracies[league]['H'] + accuracies[league]['D'] + accuracies[league]['A'] for league in accuracies])

fig, ax = plt.subplots()
bar_width = 0.2
index = np.arange(len(leagues_names))

for i in range(len(labels)):
    ax.bar(index + i * bar_width, accuracies_list[:, i], bar_width, label=labels[i])
    for j, value in enumerate(accuracies_list[:, i]):
        ax.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax.set_xlabel('Championnat')
ax.set_ylabel('Précision')
ax.set_title('Précision des prédictions par type de résultat et par championnat')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(leagues_names)
ax.legend()

plt.xticks(rotation=45)
plt.show()

# Affichage matrices de confusion
for i in range(len(leagues_names)):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Sélectionner la matrice de confusion correspondante à un championnat spécifique
    confusion_matrix = confusion_matrices_svm[i]

    # Afficher la matrice de confusion
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')

    # Afficher les valeurs dans les cases
    for j in range(len(labels)):
        for k in range(len(labels)):
            text = ax.text(k, j, confusion_matrix[j, k], ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vraie étiquette')
    ax.set_title("Matrice de confusion - SVM - " + leagues_names[i])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
"""--------------------------------------------------------MLP MAIS AVEC CETTE FOIS NOUVELLE AFFICHAGE CONFUSION MATRICE"""
accuracies = {'Ligue 1': {'H': [], 'D': [], 'A': []},
              'Bundesliga': {'H': [], 'D': [], 'A': []},
              'Liga': {'H': [], 'D': [], 'A': []},
              'Serie A': {'H': [], 'D': [], 'A': []},
              'Premier League': {'H': [], 'D': [], 'A': []}}

confusion_matrices_svm = []
"""
hidden_layer_sizes = [
    (100,),
    (100, 50),
    (50, 50),
    (100, 100, 50),
    (50, 25),
]
accuracies_mlp = {'Ligue 1': {'H': [], 'D': [], 'A': []},
                  'Bundesliga': {'H': [], 'D': [], 'A': []},
                  'Liga': {'H': [], 'D': [], 'A': []},
                  'Serie A': {'H': [], 'D': [], 'A': []},
                  'Premier League': {'H': [], 'D': [], 'A': []}}
param_grid = {'hidden_layer_sizes': hidden_layer_sizes}
f1_scores = {'Ligue 1': {'H': [], 'D': [], 'A': []},
             'Bundesliga': {'H': [], 'D': [], 'A': []},
             'Liga': {'H': [], 'D': [], 'A': []},
             'Serie A': {'H': [], 'D': [], 'A': []},
             'Premier League': {'H': [], 'D': [], 'A': []}}
confusion_matrices_mlp = []
for league, data in leagues.items():
    X = data[['niv_dom', 'niv_ext', 'diff_niv', 'forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
              'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
              'moyenne_points_ext', 'diff_point']]

    X_scaled = scaler.fit_transform(X)
    y = pd.DataFrame()
    y['resultat'] = data.apply(lambda row: resultat_lettre(row), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.25, random_state=42)

    mlp_model = MLPClassifier(activation='identity', solver='sgd', max_iter=1000, random_state=42)

    grid_search = GridSearchCV(mlp_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    model = MLPClassifier(**best_params)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(str)
    y_test = y_test.astype(str)
    # on stock les différentes accuracies afin de voir nos prédictions et de les afficher
    accuracies_mlp[league]['H'].append(accuracy_score(y_test[y_test == '3'], y_pred[y_test == '3']))
    accuracies_mlp[league]['D'].append(accuracy_score(y_test[y_test == '1'], y_pred[y_test == '1']))
    accuracies_mlp[league]['A'].append(accuracy_score(y_test[y_test == '0'], y_pred[y_test == '0']))
    f1_scores[league]['H'].append(f1_score(y_test, y_pred, average='weighted'))
    f1_scores[league]['D'].append(f1_score(y_test, y_pred, average='weighted'))
    f1_scores[league]['A'].append(f1_score(y_test, y_pred, average='weighted'))
    # Calculer la matrice de confusion
    classes = ['0', '1', '3']
    confusion_mlp = confusion_matrix(y_test, y_pred, labels=classes)
    confusion_matrices_mlp.append(confusion_mlp)
    print("Ligue:", league)
    print("Meilleurs paramètres:", best_params)
    print("Exactitude sur l'ensemble de test:", test_score)
    print("------------------")

# Affichage des précisions et F1-score
labels = ['A', 'D', 'H']
leagues_names = ['Ligue 1', 'Bundesliga', 'Liga', 'Serie A', 'Premier League']
accuracies_list = np.array([accuracies_mlp[league]['H'] + accuracies_mlp[league]['D'] + accuracies_mlp[league]['A'] for league in accuracies_mlp])
f1_scores_list = np.array([f1_scores[league]['H'] + f1_scores[league]['D'] + f1_scores[league]['A'] for league in f1_scores])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# précisions par type de résultat
bar_width = 0.2
index = np.arange(len(leagues_names))

for i in range(len(labels)):
    ax1.bar(index + i * bar_width, accuracies_list[:, i], bar_width, label=labels[i])
    for j, value in enumerate(accuracies_list[:, i]):
        ax1.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax1.set_xlabel('Championnat')
ax1.set_ylabel('Précision')
ax1.set_title('Précision des prédictions par type de résultat et par championnat')
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(leagues_names)
ax1.legend()

# F1-scores par type de résultat
f1_scores_list = np.array([f1_scores[league]['H'] + f1_scores[league]['D'] + f1_scores[league]['A'] for league in f1_scores])

for i in range(len(labels)):
    ax2.bar(index + i * bar_width, f1_scores_list[:, i], bar_width, label=labels[i])
    for j, value in enumerate(f1_scores_list[:, i]):
        ax2.text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom')

ax2.set_xlabel('Championnat')
ax2.set_ylabel('F1-score')
ax2.set_title('F1-score des prédictions par type de résultat et par championnat')
ax2.set_xticks(index + bar_width)
ax2.set_xticklabels(leagues_names)
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for i in range(len(leagues_names)):
    fig, ax = plt.subplots(figsize=(6, 6))

    confusion_matrix = confusion_matrices_mlp[i]

    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')

    for j in range(len(labels)):
        for k in range(len(labels)):
            text = ax.text(k, j, confusion_matrix[j, k], ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vraie étiquette')
    ax.set_title("Matrice de confusion - MLP - " + leagues_names[i])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
"""
"""------------------------------------------------------------------------------------------------------------------

                             ON VA FAIRE MAINTENANT LES ESTIMATEURS POUR TOUT LES MATCHS 

---------------------------------------------------------------------------------------------------------------------
"""
accuracies_normal=[]
accuracies_smote=[]
accuracies_under=[]
f1_score_normal=[]
f1_score_smote=[]
f1_score_sous=[]


param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 0.01, 0.001]
}
param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2]
}
param_grid_mlp = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'max_iter': [1500],
    'random_state': [42]
}
hidden_layer_sizes = [
    (100,),
    (100, 50),
    (50, 50),
    (100, 100, 50),
    (50, 25),
    (100, 50, 25),
    (200, 100, 50),
    (150, 100, 75, 50),
    (50, 50, 50)
]
"""
param_grid_mlp = {'hidden_layer_sizes': hidden_layer_sizes}
param_grid_knn = {'n_neighbors': range(1, 200)}
X = match[['niv_dom', 'niv_ext', 'diff_niv', 'forme_dom', 'forme_ext', 'forme_diff', 'moy_but_dom',
           'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference', 'moyenne_points_dom',
           'moyenne_points_ext', 'diff_point', 'nul_dom', 'nul_ext']]
y = match.apply(lambda row: resultat_lettre(row), axis=1)

X, y = shuffle(X, y, random_state=42)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

under_sampler = RandomUnderSampler(random_state=42)
X_train_sous, y_train_sous = under_sampler.fit_resample(X_train, y_train)
# Classifier SVM
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)
best_params_svm = grid_search_svm.best_params_
best_svm = SVC(**best_params_svm)
best_svm.fit(X_train, y_train)
y_pred_svm = best_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracies_normal.append(accuracy_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='micro')
f1_score_normal.append(f1_svm)
print("F1-score SVM :", f1_svm)

# classifier SVM pour smote
svm_smote = SVC()
grid_search_svm_smote = GridSearchCV(svm_smote, param_grid_svm, cv=5)
grid_search_svm_smote.fit(X_train_smote, y_train_smote)
best_params_svm_smote = grid_search_svm_smote.best_params_
best_svm_smote = SVC(**best_params_svm_smote)
best_svm_smote.fit(X_train_smote, y_train_smote)
y_pred_svm_smote = best_svm_smote.predict(X_test)
accuracy_svm_smote = accuracy_score(y_test, y_pred_svm_smote)
accuracies_smote.append(accuracy_svm_smote)
f1_svm_smote = f1_svm = f1_score(y_test, y_pred_svm_smote, average='micro')
f1_score_smote.append(f1_svm_smote)
print("F1-score SVM SMOTE:", f1_svm_smote)

# SVM avec sous-échantillonnage
svm_under = SVC()
grid_search_svm_under = GridSearchCV(svm_under, param_grid_svm, cv=5)
grid_search_svm_under.fit(X_train_sous, y_train_sous)
best_params_svm_under = grid_search_svm_under.best_params_
best_svm_under = SVC(**best_params_svm_under)
best_svm_under.fit(X_train_sous, y_train_sous)
y_pred_svm_under = best_svm_under.predict(X_test)
accuracy_svm_under = accuracy_score(y_test, y_pred_svm_under)
accuracies_under.append(accuracy_svm_under)
f1_svm_under = f1_score(y_test, y_pred_svm_under, average='micro')
f1_score_sous.append(f1_svm_under)
print("F1-score svm under:", f1_svm_under)
# Random Forest
rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
best_params_rf = grid_search_rf.best_params_
best_rf = RandomForestClassifier(**best_params_rf)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracies_normal.append(accuracy_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='micro')
f1_score_normal.append(f1_rf)
print("F1-score RF :", f1_rf)

# rf avec sur-échantillonnage SMOTE
rf_smote = RandomForestClassifier()
grid_search_rf_smote = GridSearchCV(rf_smote, param_grid_rf, cv=5)
grid_search_rf_smote.fit(X_train_smote, y_train_smote)
best_params_rf_smote = grid_search_rf_smote.best_params_
best_rf_smote = RandomForestClassifier(**best_params_rf_smote)
best_rf_smote.fit(X_train_smote, y_train_smote)
y_pred_rf_smote = best_rf_smote.predict(X_test)
accuracy_rf_smote = accuracy_score(y_test, y_pred_rf_smote)
accuracies_smote.append(accuracy_rf_smote)
f1_rf_smote = f1_score(y_test, y_pred_rf_smote, average='micro')
f1_score_smote.append(f1_rf_smote)
print("F1-score rf SMOTE:", f1_rf_smote)

#rf sous echantillonage
rf_under = RandomForestClassifier()
grid_search_rf_under = GridSearchCV(rf_under, param_grid_rf, cv=5)
grid_search_rf_under.fit(X_train_sous, y_train_sous)
best_params_rf_under = grid_search_rf_under.best_params_
best_rf_under = RandomForestClassifier(**best_params_rf_under)
best_rf_under.fit(X_train_sous, y_train_sous)
y_pred_rf_under = best_rf_under.predict(X_test)
accuracy_rf_under = accuracy_score(y_test, y_pred_rf_under)
accuracies_under.append(accuracy_rf_under)
f1_rf_under = f1_score(y_test, y_pred_rf_under, average='micro')
f1_score_sous.append(f1_rf_under)
print("F1-score rf under   :", f1_rf_under)
#mlp normal


grid_search_mlp = GridSearchCV(MLPClassifier(), param_grid_mlp, cv=5)
grid_search_mlp.fit(X_train, y_train)

best_params_mlp = grid_search_mlp.best_params_

best_mlp = MLPClassifier(**best_params_mlp)
best_mlp.fit(X_train, y_train)
y_pred_mlp = best_mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp, average='micro')
accuracies_normal.append(accuracy_mlp)
f1_score_normal.append(f1_mlp)
print("F1-score MLP sans modification des échantillons:", f1_mlp)

# sur-échantillonnage SMOTE mlp
grid_search_mlp_smote = GridSearchCV(MLPClassifier(), param_grid_mlp, cv=5)
grid_search_mlp_smote.fit(X_train_smote, y_train_smote)
best_params_mlp_smote = grid_search_mlp_smote.best_params_
best_mlp_smote = MLPClassifier(**best_params_mlp_smote)
best_mlp_smote.fit(X_train_smote, y_train_smote)
y_pred_mlp_smote = best_mlp_smote.predict(X_test)
accuracy_mlp_smote = accuracy_score(y_test, y_pred_mlp_smote)
f1_mlp_smote = f1_score(y_test, y_pred_mlp_smote, average='micro')
accuracies_smote.append(accuracy_mlp_smote)
f1_score_smote.append(f1_mlp_smote)
print("Meilleurs paramètres pour MLP avec sur-échantillonnage SMOTE:", best_params_mlp_smote)
print("Accuracy MLP avec sur-échantillonnage SMOTE:", accuracy_mlp_smote)
print("F1-score MLP avec sur-échantillonnage SMOTE:", f1_mlp_smote)


# Sous-échantillonnage mlp
grid_search_mlp_under = GridSearchCV(MLPClassifier(), param_grid_mlp, cv=5)
grid_search_mlp_under.fit(X_train_sous, y_train_sous)
best_params_mlp_under = grid_search_mlp_under.best_params_
best_mlp_under = MLPClassifier(**best_params_mlp_under)
best_mlp_under.fit(X_train_sous, y_train_sous)
y_pred_mlp_under = best_mlp_under.predict(X_test)
accuracy_mlp_under = accuracy_score(y_test, y_pred_mlp_under)
f1_mlp_under = f1_score(y_test, y_pred_mlp_under, average='micro')
accuracies_under.append(accuracy_mlp_under)
f1_score_sous.append(f1_mlp_under)

print("Meilleurs paramètres pour MLP avec sous-échantillonnage:", best_params_mlp_under)
print("Accuracy MLP avec sous-échantillonnage:", accuracy_mlp_under)
# Classifier KNN
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)
best_params_knn = grid_search_knn.best_params_
best_knn = KNeighborsClassifier(**best_params_knn)
best_knn.fit(X_train, y_train)
y_pred_knn = best_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='micro')
accuracies_normal.append(accuracy_knn)
f1_score_normal.append(f1_knn)
print("F1-score knn sans modification des échantillons:", f1_knn)

# Classifier KNN avec sur-échantillonnage SMOTE
knn_smote = KNeighborsClassifier()
grid_search_knn_smote = GridSearchCV(knn_smote, param_grid_knn, cv=5)
grid_search_knn_smote.fit(X_train_smote, y_train_smote)
best_params_knn_smote = grid_search_knn_smote.best_params_
best_knn_smote = KNeighborsClassifier(**best_params_knn_smote)
best_knn_smote.fit(X_train_smote, y_train_smote)
y_pred_knn_smote = best_knn_smote.predict(X_test)
accuracy_knn_smote = accuracy_score(y_test, y_pred_knn_smote)
accuracies_smote.append(accuracy_knn_smote)
f1_knn_smote = f1_score(y_test, y_pred_knn_smote, average='micro')
f1_score_smote.append(f1_knn_smote)
print("F1-score knn SMOTE   :", f1_knn_smote)

# Classifier KNN avec sous-échantillonnage
knn_under = KNeighborsClassifier()
grid_search_knn_under = GridSearchCV(knn_under, param_grid_knn, cv=5)
grid_search_knn_under.fit(X_train_sous, y_train_sous)
best_params_knn_under = grid_search_knn_under.best_params_
best_knn_under = KNeighborsClassifier(**best_params_knn_under)
best_knn_under.fit(X_train_sous, y_train_sous)
y_pred_knn_under = best_knn_under.predict(X_test)
accuracy_knn_under = accuracy_score(y_test, y_pred_knn_under)
f1_knn_under = f1_score(y_test, y_pred_knn_under, average='micro')
accuracies_under.append(accuracy_knn_under)
f1_score_sous.append(f1_knn_under)
print("F1-score knn under   :", f1_knn_under)

# Modèles
models = ['SVM', 'Random Forest', 'MLP', 'KNN']

# Trois méthodes d'échantillonnage
methods = ['Sans échantillonnage', 'Sur-échantillonnage SMOTE', 'Sous-échantillonnage']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot pour la méthode "Sans échantillonnage"
rects1 = axes[0].bar(models, accuracies_normal)
axes[0].set_title('Accuracy des modèles sans modification des échantillons')
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Précision')
axes[0].set_xlabel('Modèle')
for rect in rects1:
    height = rect.get_height()
    axes[0].annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points de décalage vertical
                     textcoords="offset points",
                     ha='center', va='bottom')

# Plot pour la méthode "Sur-échantillonnage SMOTE"
rects2 = axes[1].bar(models, accuracies_smote)
axes[1].set_title('Accuracy des modèles avec sur-échantillonnage SMOTE')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('Précision')
axes[1].set_xlabel('Modèle')
for rect in rects2:
    height = rect.get_height()
    axes[1].annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points de décalage vertical
                     textcoords="offset points",
                     ha='center', va='bottom')

# Plot pour la méthode "Sous-échantillonnage"
rects3 = axes[2].bar(models, accuracies_under)
axes[2].set_title('Accuracy des modèles avec sous-échantillonnage')
axes[2].set_ylim(0, 1)
axes[2].set_ylabel('Précision')
axes[2].set_xlabel('Modèle')
for rect in rects3:
    height = rect.get_height()
    axes[2].annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points de décalage vertical
                     textcoords="offset points",
                     ha='center', va='bottom')

# Afficher les graphiques
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot pour la méthode "Sans échantillonnage"
rects1 = axes[0].bar(models, f1_score_normal)
axes[0].set_title('F1-score des modèles sans modification des échantillons')
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('F1-score')
axes[0].set_xlabel('Modèle')
for rect in rects1:
    height = rect.get_height()
    axes[0].annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points de décalage vertical
                     textcoords="offset points",
                     ha='center', va='bottom')

# Plot pour la méthode "Sur-échantillonnage SMOTE"
rects2 = axes[1].bar(models, f1_score_smote)
axes[1].set_title('F1-score des modèles avec sur-échantillonnage SMOTE')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('F1-score')
axes[1].set_xlabel('Modèle')
for rect in rects2:
    height = rect.get_height()
    axes[1].annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points de décalage vertical
                     textcoords="offset points",
                     ha='center', va='bottom')

# Plot pour la méthode "Sous-échantillonnage"
rects3 = axes[2].bar(models, f1_score_sous)
axes[2].set_title('F1-score des modèles avec sous-échantillonnage')
axes[2].set_ylim(0, 1)
axes[2].set_ylabel('F1-score')
axes[2].set_xlabel('Modèle')
for rect in rects3:
    height = rect.get_height()
    axes[2].annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points de décalage vertical
                     textcoords="offset points",
                     ha='center', va='bottom')

# Afficher les graphiques
plt.tight_layout()
plt.show()"""