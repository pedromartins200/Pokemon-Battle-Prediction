import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from IPython.display import HTML
import numpy as np
from sklearn import preprocessing

def read_csv(filename):
    df = pd.read_csv(filename)
    return df

def path_to_image_html(path):
    return '<img src="'+ path + '" style=max-height:124px;"/>'

def showHTMLDataFrame(df):
    return HTML(df.to_html(escape=False, formatters=dict(images=path_to_image_html)))

def showPokemonImage(list, index):
    pokemon = list.loc[list['Pokedex Number'] == index]

    image = pokemon['images'].values[0]
    name = pokemon['Name'].values[0]
    plt.imshow(mpimg.imread(image))
    plt.title(name)
    plt.show()

def getPokemonName(list, index):
    pokemon = list.loc[list['Pokedex Number'] == index]
    return pokemon['Name'].values[0]

def cleanDataSet(pokemon):
    pokemon.rename(columns={"#":"Pokedex Number"}, inplace=True)
    # fill column of type 2 with None, where we have empty
    pokemon['Type 2'] = pokemon['Type 2'].fillna('None')
    return pokemon

def visualizeLegendaries(pokemon):
    legendary = pokemon.loc[(pokemon['Legendary'] == True)]
    nonLegendary = pokemon.loc[(pokemon['Legendary'] == False)]

    pieChart = [nonLegendary['Name'].count(), legendary['Name'].count()]
    plt.pie(pieChart, labels=['Non Legendary', 'Legendary'], autopct='%1.0f%%', shadow=True,
                     colors=['#ff9999', '#66b3ff'])
    plt.title('Legendary proportion in Pokemon', fontsize=12)

def visualizeTypeDistribution(pokemon):
    plt.figure(figsize=(10,10))
    labels = pokemon['Type 1'].value_counts().index
    pokemon['Type 1'].value_counts().plot(kind='pie', labels=labels, autopct='%1.0f%%')

def inspectCorrelations(pokemon):

    df = pd.DataFrame(pokemon)

    correlation = df[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]
    dimensions = (14, 9)
    fig, ax = plt.subplots(figsize = dimensions)
    correlation_map = sns.heatmap(correlation.corr(), annot=True, ax = ax, fmt=".3f", cmap=sns.cm._icefire_lut)
    correlation_map.set(title = 'HeatMap of Pokemon Base Stats')

    # Fix bug of heat map being cut off
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)


def replace(data, stats_dict, type_dict):
    data['First_pokemon_stats'] = data.First_pokemon.map(stats_dict)
    data['Second_pokemon_stats'] = data.Second_pokemon.map(stats_dict)

    data['First_pokemon'] = data.First_pokemon.map(type_dict)
    data['Second_pokemon'] = data.Second_pokemon.map(type_dict)

    return data


def get_difference_stats(data):
    stats_col = ["HP_diff", "Attack_diff", "Defense_diff", 'Sp.Atk_diff', 'Sp.Def_diff', 'Speed_diff']
    diff_list = []

    for row in data.itertuples():
        diff_list.append(np.array(row.First_pokemon_stats) - np.array(row.Second_pokemon_stats))

    stats_df = pd.DataFrame(diff_list, columns=stats_col)

    data = pd.concat([data, stats_df], axis=1)

    data.drop(['First_pokemon_stats', 'Second_pokemon_stats'], axis=1, inplace=True)

    data = data.drop(['First_pokemon', 'Second_pokemon'], axis=1)

    return data


def normalize(data):
    for c in data:
        description=data[c].describe()
        data[c]=(data[c]-description['min'])/(description['max']-description['min'])
    return data

def createDictionaries(pokemon_combats):
    type_dic = pokemon_combats.iloc[:, 0:4]
    type_dic = type_dic.drop("Name", axis=1)
    stats_dic = pokemon_combats.drop(['Name', 'Type 1', 'Type 2', 'Generation', 'Legendary'], axis=1)

    type_dict = type_dic.set_index('Pokedex Number').T.to_dict('list')
    stats_dict = stats_dic.set_index('Pokedex Number').T.to_dict('list')

    return type_dict, stats_dict
