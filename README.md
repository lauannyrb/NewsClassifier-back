## NewsClassifier

### Classificador de Notícias com IA e Recomendação Personalizada

O NewsClassifier é uma aplicação que utiliza inteligência artificial para classificar automaticamente notícias em diferentes categorias e fornecer recomendações personalizadas.

### Acesse o site clicando aqui: [NewsClassifier](https://newsclassifier-py.streamlit.app/)

**Funcionalidades Principais:**

* **Classificação Automática:** Analisa a machete da notícia e a classifica em categorias relevantes (política, esportes, tecnologia, etc.).
* **Recomendações Personalizadas:** Sugere outras notícias com base na categoria da notícia classificada.
* **Interface Intuitiva:** Permite que os usuários insiram facilmente o título da notícia e visualizem os resultados da classificação e as recomendações.

**Tecnologias Utilizadas:**

* **Framework Web:** Streamlit (Python)
* **Back-end:** Python
* **Modelo de Machine Learning:** Naive Bayes (com TF-IDF para vetorização de texto)
* **API de Notícias:** NewsAPI
* **Outras Bibliotecas:** Pandas, Scikit-learn, Requests

**Como Executar o Projeto:**

1. **Clone o Repositório:**
   ```bash
   git clone https://github.com/lauannyrb/NewsClassifier-back.git
   ```
2. **Crie e Ative o Ambiente Virtual (venv):**

   a. **Crie o ambiente:**
      ```bash
      python -m venv venv
      ```

   b. **Ative o ambiente:**

      * **No Linux/macOS:**
         ```bash
         source venv/bin/activate
         ```

      * **No Windows:**
         ```bash
         venv\Scripts\activate
         ```
3. **Instale as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Inicie o Aplicativo Streamlit:**
   ```bash
   streamlit run app.py
   ```
5. **Acesse a Aplicação:**
   * Abra seu navegador e acesse `http://localhost:8501`.
6. **Insira a Notícia:**
   * Cole o título da notícia na área de texto fornecida.
7. **Clique em "Classificar":**
   * O aplicativo classificará a notícia e exibirá a categoria e as recomendações.

**Observações:**

* Certifique-se de ter uma chave de API válida da NewsAPI e substitua o valor em `api_key` no código.
* O arquivo `news_articles.csv` deve estar na mesma pasta do código `app.py` ou o caminho deve ser ajustado.

**Desenvolvedores:**

* Lauanny Rodrigues ([https://github.com/lauannyrb](https://github.com/lauannyrb))
* Jonas Oliveira ([https://github.com/Jonas-Oliveira-12](https://github.com/Jonas-Oliveira-12))

