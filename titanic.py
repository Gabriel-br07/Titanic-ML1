import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carrega os dados de treinamento e teste a partir de arquivos CSV
treino = pd.read_csv("train.csv")
teste = pd.read_csv("test.csv")

# Extrai os valores de PassengerId para o conjunto de teste
teste_ids = teste["PassengerId"]

# Configurações para exibição no Pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Função para limpar e pré-processar os dados
def limpar(dados):
    # Remove colunas desnecessárias
    dados = dados.drop(["Ticket", "Fare", "Cabin", "Name", "PassengerId"], axis=1)

    # Preenche valores ausentes na coluna 'Age' com a idade média
    idade_media = dados["Age"].mean()
    dados["Age"].fillna(idade_media, inplace=True)

    # Preenche valores ausentes na coluna 'Embarked' com 'U' (Desconhecido)
    dados.Embarked.fillna("U", inplace=True)

    # Codifica strings categóricas ('Sex' e 'Embarked') em valores numéricos
    le = LabelEncoder()
    dados["Sex"] = le.fit_transform(dados["Sex"])
    dados["Embarked"] = le.fit_transform(dados["Embarked"])

    return dados

# Limpa e pré-processa os dados de treinamento e teste
treino = limpar(treino)
teste = limpar(teste)

# Aplicação do Modelo de Regressão Logística
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

# Separa as características (x) e a variável alvo (y)
x = treino.drop(["Survived"], axis=1)
y = treino["Survived"]

# Define uma semente aleatória para reprodutibilidade
SEED = 311
np.random.seed(SEED)

# Cria um modelo de Regressão Logística
modelo = LogisticRegression()

# Aplica validação cruzada com 10 dobras (10-fold cross-validation)
resultado = cross_validate(modelo, x, y, cv=10, return_train_score=False)
media = resultado['test_score'].mean()
desvio_padrao = resultado['test_score'].std()

# Ajusta o modelo aos dados de treinamento
modelo.fit(x, y)

# Faz previsões nos dados de teste
result_final = modelo.predict(teste)

# Cria um DataFrame com as colunas PassengerId e Survived para submissão
df = pd.DataFrame({"PassengerId": teste_ids.values, "Survived": result_final})

# Salva as previsões em um arquivo CSV
df.to_csv("resultado.csv", index=False)
