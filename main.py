import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
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
#meme chose des ligues
ligue = pd.read_sql("SELECT * FROM League;", conn)

match=pd.read_sql("SELECT * FROM Match;", conn)

#je récupère la bdd des joueurs
joueur_attribue = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
#dommage que les stats de tirs pour, contre, possesion sont pas complete, je prefère donc les enlever (seulement 20% des matchs ont ces stats)
#j'enlève également la grosse majorité des colonnes qui ne servent pas à grands chose), suppression de toutes les colonnes contenant
#les cotes des bets car l'objectif c'est de predire à partir de statistiques, pas à partir des cotes des matchs

reduction_match=['id','country_id','league_id','season','stage','date','match_api_id','home_team_api_id','away_team_api_id'
    ,'home_team_goal','away_team_goal','home_player_1','home_player_2','home_player_3','home_player_4','home_player_5','home_player_6'
                 ,'home_player_7','home_player_8','home_player_9','home_player_10','home_player_11','away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6'
                 ,'away_player_7','away_player_8','away_player_9','away_player_10','away_player_11']
match=match[reduction_match]
#match.dropna(inplace=True)
match = match[match['country_id'] == 4769]
#print(match)
#objectif créé une nouvelle data frame avec les colonnes suivantes : état de forme de l'équipe avec le nb de victoire sur
#les 10 derniers match, le nombre de match nul, le nombre de défaite ou alors juste une colonne avec le nombre de points pris sur les 0 derniers match
#ensuite un tableau avec la moyenne du rank des joueurs présent sur le terrain
#nombre de point en confrontation direct sur les 5 derniers match
#nombre de but pour, nombre de but contre
"""
listes des fonctions : 
 resultat(match,id_team) : resultat du match a partir de l'id d'une equipe,on l'utilisera pour calculer la forme d'une eq
 
 diff_form(match,match_id): fera la difference de forme entre 2 equipes
 
 niv_dom(match),niv_ext(match),diff_niv (match) : fais la moyenne de l'eq a dom, a l'ext et on prends la difference
 
 but_marque(match,id_team),but_encaisse(match,id_team): permet d'obtenir les but enc et marq d'une eq dans un match
 
 nombre_but_match(match,match_id): permet d'avoir le nombre de but par match de l'eq a dom marquer,encaissé
                                   et de l'equipe exterieur, renvoie également la difference
 
"""

#résultat du match afin de calculer les points, sert pour la fonction d'après pour obtenir état de forme des équipes
def resultat(match,id_team):
    but_dom=match.home_team_goal
    but_ext=match.away_team_goal
    #print(but_dom)
    #print(but_ext)
    if(id_team==match.home_team_api_id):
        if(but_dom==but_ext):
            return 1
        if(but_dom>but_ext):
            return 3
        if(but_dom<but_ext):
            return 0
    else :
        if(but_dom==but_ext):
            return 1
        if(but_dom<but_ext):
            return 3
        if(but_dom>but_ext):
            return 0
#du coup val pos, forme en faveur de l'équipe a domicile, val neg opposé
def diff_form(match,match_id):
    date_match = match_id['date']
    equipe_dom_id=match_id.home_team_api_id
    equipe_ext_id=match_id.away_team_api_id
    #print(equipe_dom_id)
    #print(equipe_ext_id)
    derniers_matchs_dom = match[(match.home_team_api_id == equipe_dom_id) | (match.away_team_api_id == equipe_dom_id)]
    derniers_matchs_dom = derniers_matchs_dom[derniers_matchs_dom['date'] < date_match].tail(5)
    points_dom = sum(derniers_matchs_dom.apply(lambda x: resultat(x, equipe_dom_id), axis=1))
    # Calcul des points pour l'équipe à l'extérieur
    derniers_matchs_ext = match[(match['home_team_api_id'] == equipe_ext_id) | (match['away_team_api_id'] == equipe_ext_id)]
    derniers_matchs_ext = derniers_matchs_ext[derniers_matchs_ext['date'] < date_match].tail(5)
    #print(derniers_matchs_ext)
    points_ext = sum(derniers_matchs_ext.apply(lambda x: resultat(x, equipe_ext_id), axis=1))
    #print("résultat: ")
    #print(match_id.home_team_goal,match_id.away_team_goal)
    diff_point=points_dom-points_ext
    return diff_point,points_dom,points_ext
#cette fonction va faire en sorte qu'on est la moyenne du niveau de l'équipe à domicile
def niv_dom(match):
    date_match = match['date']
    somme=0
    coef=1.3
    joueur_stat=pd.DataFrame()
    match_id=match.match_api_id
    home_team=['home_player_1','home_player_2','home_player_3','home_player_4','home_player_5','home_player_6'
                 ,'home_player_7','home_player_8','home_player_9','home_player_10','home_player_11']
    for joueur in home_team :
        joueur_id=match[joueur]
        joueur_stat = joueur_attribue[(joueur_attribue['player_api_id'] == joueur_id) & (joueur_attribue['date'] < date_match)]
        joueur_stat = joueur_stat.sort_values(by='date', ascending=False).head(1)
        if joueur_stat.empty:
            overall_rating = 40  # Valeur par défaut si aucune correspondance n'est trouvée
        else:
            overall_rating = joueur_stat['overall_rating'].iloc[0]
        #print(overall_rating.iloc[0])
        somme=somme+overall_rating
        #print(somme)
    moy=somme/11
    moy=moy*1.10
    return moy
def niv_ext(match):
    date_match = match['date']
    somme=0
    joueur_stat=pd.DataFrame()
    match_id=match.match_api_id
    away_team=['away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6'
    , 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11']
    for joueur in away_team :
        joueur_id=match[joueur]
        joueur_stat = joueur_attribue[(joueur_attribue['player_api_id'] == joueur_id) & (joueur_attribue['date'] < date_match)]
        joueur_stat = joueur_stat.sort_values(by='date', ascending=False).head(1)
        if joueur_stat.empty:
            overall_rating = 40  # Valeur par défaut si aucune correspondance n'est trouvée
        else:
            overall_rating = joueur_stat['overall_rating'].iloc[0]
        somme=somme+overall_rating
        #print(somme)
    moy=somme/11
    return moy
#je m'interresse a la difference de niveau car finalement c'est cela qui risque d'impacter le plus le résultat
def diff_niv (match):
    difference=niv_dom(match)-niv_ext(match)
    return difference

#fonction qui va dire le résultat, sauf que je vais dire V, D ,N
def resultat_lettre(match):
    but_dom=match.home_team_goal
    but_ext=match.away_team_goal
    #print(but_dom)
    #print(but_ext)
    if(but_dom==but_ext):
        return 1
    if(but_dom>but_ext):
        return 3
    if(but_dom<but_ext):
        return 0
#pour calculer le nombre de but d'une equipe je vais cree une fonction qui va correctement renvoyer le nombre de but
def but_marque(match,id_team):
    if(match.home_team_api_id==id_team):
        return match.home_team_goal
    else : return match.away_team_goal
#meme principe mais avec les but encaissé
def but_encaisse(match,id_team):
    if(match.home_team_api_id==id_team):
        return match.away_team_goal
    else: return match.home_team_goal
#ceci va renvoyer 4 nombre, la moyenne du nombre de but marque,celle encaissé des deux équipes
def nombre_but_match(match,match_id):
    date_match = match_id['date']
    equipe_dom_id = match_id.home_team_api_id
    equipe_ext_id = match_id.away_team_api_id
    season_match=match_id.season
    #print(match_id.stage)
    total_match_saison_dom = match[((match.home_team_api_id == equipe_dom_id) | (match.away_team_api_id == equipe_dom_id))
                            & (match.season==season_match)]
    total_match_saison_dom = total_match_saison_dom[total_match_saison_dom['date'] < date_match]
    total_match_saison_dom['but_M'] = total_match_saison_dom.apply(lambda row: but_marque(row, equipe_dom_id), axis=1)
    somme_but_dom = total_match_saison_dom['but_M'].sum()
    total_match_saison_dom['but_E'] = total_match_saison_dom.apply(lambda row: but_encaisse(row, equipe_dom_id), axis=1)
    somme_but_enc_dom = total_match_saison_dom['but_E'].sum()
    nombre_matchs_dom = len(total_match_saison_dom)
    #print(somme_but_dom)
    #print(somme_but_enc_dom)
    #print(total_match_saison_dom)
    if(nombre_matchs_dom!=0):
        somme_but_dom=somme_but_dom/nombre_matchs_dom
        somme_but_enc_dom=somme_but_enc_dom/nombre_matchs_dom
    #print(somme_but_dom)
    #print(somme_but_enc_dom)
    total_match_saison_ext = match[
        ((match.home_team_api_id == equipe_ext_id) | (match.away_team_api_id == equipe_ext_id))
        & (match.season == season_match)]
    total_match_saison_ext = total_match_saison_ext[total_match_saison_ext['date'] < date_match]

    total_match_saison_ext['but_M'] = total_match_saison_ext.apply(lambda row: but_marque(row, equipe_ext_id), axis=1)
    somme_but_ext = total_match_saison_ext['but_M'].sum()
    total_match_saison_ext['but_E'] = total_match_saison_ext.apply(lambda row: but_encaisse(row, equipe_ext_id), axis=1)
    somme_but_enc_ext = total_match_saison_ext['but_E'].sum()
    nombre_matchs_ext = len(total_match_saison_ext)
    if(nombre_matchs_ext!=0):
        somme_but_ext = somme_but_ext / nombre_matchs_ext
        somme_but_enc_ext = somme_but_enc_ext / nombre_matchs_ext
    #print(somme_but_ext)
    #print(somme_but_enc_ext)
    if(nombre_matchs_dom == 0 | nombre_matchs_ext == 0):
        return 0,0,0,0,0
    difference=somme_but_dom+somme_but_enc_ext-somme_but_ext-somme_but_enc_dom
    return somme_but_dom,somme_but_enc_dom,somme_but_ext,somme_but_enc_ext,difference
nombre_but_match(match,match.iloc[25])


def afficher_nombre_match_par_saison(df):
    saison_counts = df['season'].value_counts().sort_index()

    for saison, count in saison_counts.items():
        print(f"Saison {saison}: {count} match(s)")


# Utilisation de la fonction
afficher_nombre_match_par_saison(match)
def point_saison(match,match_id):
    date_match = match_id['date']
    equipe_dom_id = match_id.home_team_api_id
    equipe_ext_id = match_id.away_team_api_id
    season_match = match_id.season
    #on prends les amtch de la saison
    total_match_saison_dom = match[
        ((match.home_team_api_id == equipe_dom_id) | (match.away_team_api_id == equipe_dom_id))
        & (match.season == season_match) & (match.date < date_match)]

    #nb de points de l'equipe à dom
    points_dom = sum(total_match_saison_dom.apply(lambda x: resultat(x, equipe_dom_id), axis=1))
    nombre_matchs_dom = len(total_match_saison_dom)
    if(nombre_matchs_dom==0):
        return 0,0,0
    else:
        moyenne_points_dom = points_dom / nombre_matchs_dom
    # meme chose on prends les match de la saison avant la date
    total_match_saison_ext = match[
        ((match.home_team_api_id == equipe_ext_id) | (match.away_team_api_id == equipe_ext_id))
        & (match.season == season_match) & (match.date < date_match)]

    # nb points equipe à ext
    points_ext = sum(total_match_saison_ext.apply(lambda x: resultat(x, equipe_ext_id), axis=1))
    nombre_matchs_ext = len(total_match_saison_ext)
    if(nombre_matchs_ext==0):
        return 0,0,0
    else :
        moyenne_points_ext = points_ext / nombre_matchs_ext
        diff_point = moyenne_points_dom - moyenne_points_ext
    return diff_point, moyenne_points_dom, moyenne_points_ext
def opti_point(row):
    diff_point, moyenne_points_dom, moyenne_points_ext = point_saison(match, row)
    return diff_point, moyenne_points_dom, moyenne_points_ext


#PERMET DOPTIMISER AU LIEU DE FAIRE APPLIQUER LA FONCTION 3 FOIS
def opti_forme(row):
    forme_dom, forme_ext, forme_diff = diff_form(match, row)
    return forme_dom, forme_ext, forme_diff
def opti_but(row):
    moy_but_dom, moy_but_enc_dom, moy_but_ext, moy_but_enc_ext, difference = nombre_but_match(match, row)
    return moy_but_dom, moy_but_enc_dom, moy_but_ext, moy_but_enc_ext, difference

start_time = time.time()
match[['forme_dom', 'forme_ext', 'forme_diff']] = match.apply(lambda  row: opti_forme(row), axis=1, result_type='expand')

print("Les features 'forme_dom', 'forme_ext' et 'forme_diff' ont été calculées en %s secondes" % (time.time() - start_time))

start_time = time.time()
match['niv_dom'] = match.apply(lambda row: niv_dom(row), axis=1)
print("La feature 'niv_dom' a été calculée en %s secondes" % (time.time() - start_time))

start_time = time.time()
match['niv_ext'] = match.apply(lambda row: niv_ext(row), axis=1)
print("La feature 'niv_ext' a été calculée en %s secondes" % (time.time() - start_time))

start_time = time.time()
match['diff_niv'] = match.apply(lambda row: diff_niv(row), axis=1)
print("La feature 'diff_niv' a été calculée en %s secondes" % (time.time() - start_time))
start_time = time.time()
match[['diff_point', 'moyenne_points_dom', 'moyenne_points_ext']] = match.apply(lambda row :opti_point(row), axis=1, result_type='expand')

print("Les features 'diff_point', 'moyenne_points_dom' et 'moyenne_points_ext' ont été calculées en %s secondes" % (time.time() - start_time))
start_time = time.time()
match[['moy_but_dom', 'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext', 'difference']] = match.apply(lambda row :opti_but(row), axis=1, result_type='expand')

print("Les features 'moy_but_dom', 'moy_but_enc_dom', 'moy_but_ext', 'moy_but_enc_ext' et 'difference' ont été calculées en %s secondes" % (time.time() - start_time))
print(match.iloc[44])


y=pd.DataFrame()
y['resultat'] = match.apply(lambda row: resultat_lettre(row), axis=1)
X=match[['niv_dom','diff_niv','niv_ext','forme_dom','forme_ext','forme_diff','moy_but_dom','moy_but_enc_dom','moy_but_ext',
         'moy_but_enc_ext','difference','moyenne_points_dom','moyenne_points_ext','diff_point']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
#-------------------------------------------------------------------------------------------------------------------
# diviser les donnée entre les données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.25, random_state=42)
# valeurs de k à tester
param_grid = {'n_neighbors': range(1, 200)}

knn = KNeighborsClassifier()

# on ba essayer de trouver le meilleur k par cross validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# on trouve le meilleur k
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

#on utilise le k trouver precedement
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

test_score = best_knn.score(X_test, y_test)

print("k optimal :", best_k)
print("Score de validation croisée:", best_score)
print("Score sur l'ensemble de test:", test_score)


param_grid = {'n_estimators': [50, 100, 150, 200, 250]}

rf = RandomForestClassifier()

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

#pour trouver le n_estimators adequats
best_estimators = grid_search.best_params_['n_estimators']
best_score = grid_search.best_score_
#on fait la demarche avec nos valeur
best_rf = RandomForestClassifier(n_estimators=best_estimators)
best_rf.fit(X_train, y_train)

test_score = best_rf.score(X_test, y_test)

print("Meilleur nombre d'estimateurs trouvé:", best_estimators)
print("Score de validation croisée:", best_score)
print("Score sur l'ensemble de test:", test_score)
