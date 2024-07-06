import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Carregar os dados do arquivo CSV (substitua pelo caminho correto)
csv_file = "./news_articles.csv"
df = pd.read_csv(csv_file)

# Exemplo de dados de treinamento (textos e categorias)
train_data = list(zip(df['text'], df['category']))
texts, labels = zip(*train_data)

# Criar um pipeline para vetorizar os textos e treinar o classificador Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
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

# Chave da API da NewsAPI (substitua pela sua chave real)
api_key = '246fa7e8872b4475af4dacca634b8dfe' 

def main():
    
    st.set_page_config(page_title="NewsClassifier", page_icon=":newspaper:")
  
    st.html("<h1 style='text-align: center; color: #007BFF;'>NewsClassifier</h1>")
    st.html("<h3 style='text-align: center;'>Classifique notícias e receba sugestões personalizadas!</h3>")


    with st.form("news_form"):
        manchete = st.text_area("Insira a manchete (título) da notícia:", height=150, placeholder="Digite a manchete aqui...")
        submitted = st.form_submit_button("CLASSIFICAR", use_container_width=True)


        if submitted and manchete:
            with st.spinner("Classificando e buscando recomendações..."):
                result = process_news_and_recommend(api_key, manchete)

            categoria = result["categoria"]
            recomendacoes = result["recomendacoes"]

            st.markdown(f"<h4 style='text-align: center; color: #28a745;'>Categoria da notícia: {categoria}</h4>", unsafe_allow_html=True)

            if recomendacoes:
                st.markdown("<h5 style='text-align: center; color: #17a2b8;'>Recomendações de notícias:</h5>", unsafe_allow_html=True)
                for title, url in recomendacoes[:3]:
                    st.markdown(f"- [{title}]({url})", unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align: center; color: #ffc107;'>Não encontramos recomendações para esta categoria no momento.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
