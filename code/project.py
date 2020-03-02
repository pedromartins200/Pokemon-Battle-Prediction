from sklearn.tree import DecisionTreeClassifier

from helpers import *
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import warnings


warnings.filterwarnings("ignore")

# read raw csv file
pokemon = read_csv("pokemon.csv")

combats = read_csv("combats.csv")

combat_tests = read_csv("tests.csv")

# clean some stuff
pokemon = cleanDataSet(pokemon)

pokemon_combats_df = pokemon.copy()

# insert image path as a new column
# some images have incorrect name... but oh well couldnt find a better image set
pokemon.insert(12, "images", "images/" + pokemon['Name'] + ".png")

# Find pokemon that didnt win a single combat !
total_wins = combats.Winner.value_counts()
numberOfWins = combats.groupby('Winner').count()
countByFirst = combats.groupby('Second_pokemon').count()
countBySecond = combats.groupby('First_pokemon').count()

losing_pokemon = np.setdiff1d(countByFirst.index.values, numberOfWins.index.values) - 1
losing_pokemon = pokemon.iloc[losing_pokemon[0],]
#showPokemonImage(pokemon, losing_pokemon.values[0])


# List pokemons that won the most
numberOfWins = numberOfWins.sort_index()
numberOfWins['Total Fights'] = countByFirst.Winner + countBySecond.Winner
numberOfWins['Win Percentage'] = numberOfWins.First_pokemon / numberOfWins['Total Fights']

final_result = pd.merge(pokemon, numberOfWins, left_on="Pokedex Number", right_index=True, how='left')

print(final_result.describe().to_string())

print(final_result[np.isfinite(final_result['Win Percentage'])].sort_values(by = ['Win Percentage'], ascending=False).head(10))


# Show the types that win the most
print(final_result.groupby('Type 1').agg({"Win Percentage": "mean"}).sort_values(by = "Win Percentage", ascending=False))


# LETS DO SOME MACHINE LEARNING NOW

# Combat winner prediction
combats.Winner[combats.Winner == combats.First_pokemon] = 0
combats.Winner[combats.Winner == combats.Second_pokemon] = 1

type_dict, stats_dict = createDictionaries(pokemon_combats_df)


combats = replace(combats, stats_dict, type_dict)
train_df = get_difference_stats(combats)
train_df = normalize(train_df)

y_train_full = train_df['Winner']
x_train_full = train_df.drop('Winner', axis=1)

# Undersampled data

undersample = train_df.sample(200)

y_train_full_undersampled = undersample['Winner']
x_train_full_undersampled = undersample.drop('Winner', axis=1)

test_size = 0.2
x_train, x_cv, y_train, y_cv = train_test_split(x_train_full, y_train_full,
                                                test_size=test_size)

# Type prediction

types = list(pokemon['Type 1'].unique())
test_data = pd.DataFrame(columns=pokemon.columns)

frac = 0.1
for type in types:
    test_data = test_data.append((pokemon[pokemon['Type 1'] == type]).sample(frac=frac))

train_data = pokemon[~pokemon['Name'].isin(test_data['Name'])]

# scale features
max_abs_scaler = preprocessing.MaxAbsScaler()

# 4-10: from Total to Speed
x_train_type = max_abs_scaler.fit_transform(train_data.iloc[:, 4:10])
x_test_type = max_abs_scaler.fit_transform(test_data.iloc[:, 4:10])

array_train = train_data.values
array_test = test_data.values

y_train_type = array_train[:, 2]
y_test_type = array_test[:, 2]

algorithms = {'naive bayes': GaussianNB(),
            'log reg': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'random forest': RandomForestClassifier(n_estimators=100)}


for name, algorithm in algorithms.items():
    start_time = time.time()
    model = algorithm.fit(x_train, y_train)
    pred = model.predict(x_cv)
    print('Combat prediction, Accuracy of {}:'.format(name), accuracy_score(pred, y_cv), " in", time.time() - start_time, "seconds")
    start_time = time.time()
    model_type = algorithm.fit(x_train_type, y_train_type)
    pred_type = model_type.predict(x_test_type)
    print('Type prediction, Accuracy of {}:'.format(name), accuracy_score(pred_type, y_test_type), " in", time.time() - start_time, "seconds")





# Random forest appeared to be the best one
tests = read_csv("tests.csv")
prediction_df = tests.copy()
test_df = replace(tests.copy(), stats_dict, type_dict)
test_df = get_difference_stats(test_df)
test_df = normalize(test_df)


classifier = RandomForestClassifier(n_estimators=100)
model = classifier.fit(x_train_full, y_train_full)
prediction = model.predict(test_df)


# Oversampling legendaries with smote

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(x_train_full_undersampled,y_train_full_undersampled, test_size = 0.30)

sm = SMOTE(random_state=12, sampling_strategy=1.0)
X_train_res, y_train_res = sm.fit_sample(X_train_smote, y_train_smote)

model_with_smote = classifier.fit(X_train_res, y_train_res)
pred_with_smote = model_with_smote.predict(X_test_smote)

print("Accuracy with random forest using smote ",  accuracy_score(pred_with_smote, y_test_smote))

############### Visualizing a random tree ##############

# This generates a .dot file. Afterwards use a online .dot visualizer to convert to .svg or .png

#estimator = model.estimators_[5]

#export_graphviz(estimator,
                #out_file='tree.dot',
                #feature_names = x_train_full.columns,
                #rounded = True, proportion = False,
                #precision = 2, filled = True)



#for i in range(10):

    #if prediction[i] == 0:
       # winner = tests.loc[i]['First_pokemon']
    #else:
       # winner = tests.loc[i]['Second_pokemon']

    #print(tests.loc[i]['First_pokemon'], "(",getPokemonName(pokemon,tests.loc[i]['First_pokemon']), ") x ",
         # tests.loc[i]['Second_pokemon'], "(",getPokemonName(pokemon,tests.loc[i]['Second_pokemon']), ")"  ", Winner = ", winner)


########## SOME PRELIMINARY DATA VISUALIZATION ##########

# Most important features of the model

#feat_importances = pd.Series(model.feature_importances_, index=x_train_full.columns)
#feat_importances.nlargest(10).plot(kind='barh')


#visualizeLegendaries(pokemon)

#plt.show()


# 2. Proportion of each type
# visualizeTypeDistribution(pokemon)

# 3. View correlation of each stat
# inspectCorrelations(pokemon)

# plt.show()
