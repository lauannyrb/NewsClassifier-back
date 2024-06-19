import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import os
import pandas as pd

# Obter o caminho completo para o arquivo CSV
current_dir = os.path.dirname(__file__)  # Diretório atual do script Python
csv_file = os.path.join(current_dir, '/home/jonas/Documentos/Projetos/NewsClassifier-back/news_articles.csv')

# Carregar os dados do arquivo CSV
df = pd.read_csv(csv_file)


# Exemplo de dados de treinamento (textos e categorias)
train_data = list(zip(df['text'], df['category']))

# Separar os dados de treinamento em textos e labels
texts, labels = zip(*train_data)

# Criar um pipeline para vetorizar os textos e treinar o classificador Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Treinar o modelo
model.fit(texts, labels)

# Função para categorizar uma nova notícia
def categorize_news(model, news_title):
    prediction = model.predict([news_title])
    return prediction[0]

# Função para buscar notícias da NewsAPI
def get_news(api_key, query, language='pt'):
    url = f'https://newsapi.org/v2/everything?q={query}&language={language}&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('articles', [])
    return [(article['title'], article['url']) for article in articles]

# Função principal que será chamada pelo front-end
def process_news_and_recommend(api_key, news_title):
    # Classificar a notícia inserida
    category = categorize_news(model, news_title)
    
    # Buscar notícias recomendadas com base na categoria
    recommendations = get_news(api_key, category)
    
    # Retornar a categoria e as recomendações
    return {
        "categoria": category,
        "recomendacoes": recommendations
    }

# Exemplo de chamada da função (substitua 'SUA_CHAVE_API' pela sua chave real da NewsAPI)
api_key = '246fa7e8872b4475af4dacca634b8dfe'
news_title = input("Digite o título da notícia: ")
result = process_news_and_recommend(api_key, news_title)

# Exibir o resultado
print(f'Título: {news_title}\nCategoria: {result["categoria"]}')
print("\nRecomendações de notícias:")
for title, url in result["recomendacoes"]:
    print(f'- {title}: {url}')
