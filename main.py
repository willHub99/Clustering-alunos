import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus


train_df = pd.read_csv('alunos_engcomp.csv', delimiter=';', decimal=',')

print("==============================")
print("ANTES DO PRE-PROCESSAMENTO")
print("==============================")
print(train_df.head())
# M -> 0 e F -> 1 
train_df['sexo'] = train_df['sexo'].apply(lambda sex: 0 if sex == 'M' else 1)
# publica -> 0 e privada -> 1
train_df['escola'] = train_df['escola'].apply(lambda esc: 0 if esc == 'publica' else 1)
#transforma os elementos da coluna periodo de string para int
train_df['periodo'] = train_df['periodo'].apply(lambda x: int(x))

print("==============================")
print("DEPOIS DO PRE-PROCESSAMENTO")
print("==============================")
print(train_df.head())

#cria um vetor com as colunas utilizadas como entrada do modelo
columns = ['coeficiente', 'periodo', 'escola', 'enem']
x = train_df[list(columns)].values
#conjunto de saída do modelo
y = train_df["sexo"].values

#separa os dados de entrada e saida (treino e teste)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 3)

#cria os pesos árvore de decisão
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
#cria o modelo com os dados de treino
model = clf.fit(X_train,Y_train)

#cria e exporta a árvore construída
dot_data = tree.export_graphviz(model, out_file=None,class_names=columns, feature_names=columns, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('alunos.png')

#verifica a acrácia do modelo
Y = model.predict(X_test)

print("==============================")
print("NIVEL DE ACURACIA")
print("==============================")
precisao = accuracy_score(Y_test, Y)
print(f"Acruracia do modelo {precisao}")
