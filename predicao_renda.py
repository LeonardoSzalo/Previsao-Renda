import streamlit as st
import intro
import demografico_streamlit  # Importa o primeiro script como um módulo
import predicao_renda  # Importa o segundo script como um módulo
import predicao_renda_binaria # Importa o terceiro script como um módulo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    roc_curve
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def run():
    # Configurações iniciais
    color_palette = ['#023047', '#e85d04', '#0077b6', '#0096c7', '#ff9c33']
    sns.set_palette(sns.color_palette(color_palette))
    pd.set_option('display.max_columns', None)

    # Carregar o seu DataFrame renda
    renda = pd.read_csv(r"C:\Users\leosz\Desktop\Previsao_renda_streamlit\previsao_de_renda.xls")
    
    # Pré-processamento
    renda.drop('Unnamed: 0', axis=1, inplace=True)
    renda = renda.fillna(0)
    def categorizar_tempo_emprego(tempo):
        if 0 <= tempo < 0.1:
            return '0 anos'
        elif 0.1 < tempo < 5:
            return '1 a 4.99 anos'
        elif 5 < tempo < 10:
            return '5 a 10 anos'
        elif 10 < tempo < 15:
            return '10 a 15 anos'
        elif 15 < tempo:
            return '15 anos ou mais'
        return tempo

    renda['tempo_emprego_categorico'] = renda['tempo_emprego'].apply(categorizar_tempo_emprego)

    def categorize_filhos(qtd):
        if qtd == 0:
            return '0 filhos'
        elif 1 <= qtd <= 2:
            return '1-2 filhos'
        else:
            return '3 ou mais filhos'

    renda['categoria_filhos'] = renda['qtd_filhos'].apply(categorize_filhos)

    bins = [0, 25, 30, 40, 50, 60, 100]  # Definindo os limites das categorias
    labels = ['<25 anos', '<=25 e <30 anos', '<=30 e <40 anos', '<=40 e <50 anos', '<=50 e <=60 anos', '>60 anos']  # Nomes das categorias
    renda['faixa_idade'] = pd.cut(renda['idade'], bins=bins, labels=labels, right=False)

    dummies = renda.copy()
    dummies.drop(['data_ref', 'id_cliente'], axis=1, inplace=True)
    dummies = pd.get_dummies(dummies, drop_first=True)
    
    # Transformação da renda em log
    dummies['renda_log'] = np.log(dummies['renda'])
    
    df_numerico = dummies.select_dtypes(include=['float64', 'int64'])
    df_numerico.drop(['renda', 'renda_log'], axis=1, inplace=True)
    
    # Padronizando os dados
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numerico)
    
    # Aplicando PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    
    # Convertendo para DataFrame
    df_pca = pd.DataFrame(data=df_pca, columns=['Componente 1', 'Componente 2'])
    
    df_final = pd.concat([dummies.select_dtypes(include='bool').reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)
    
    # Separando recursos e alvo
    X = df_final  # Recursos
    y = dummies['renda_log']  # Target
    
    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinamento do modelo Lasso
    model_lasso = Lasso(alpha=0.01, max_iter=10000)
    model_lasso.fit(X_train, y_train)
    
    # Previsões
    y_train_pred = model_lasso.predict(X_train)
    y_test_pred = model_lasso.predict(X_test)
    
    # Avaliação do desempenho
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Exibindo métricas no Streamlit
    st.title("Regressão Linear - Avaliação de Desempenho")
    st.write(f'Treinamento MSE: {train_mse:.2f}, R²: {train_r2:.2f}')
    st.write(f'Teste MSE: {test_mse:.2f}, R²: {test_r2:.2f}')
    
    # Verificação de Overfitting
    if train_r2 - test_r2 > 0.1:
        st.warning("Possível overfitting detectado.")
    
    # Análise de Resíduos
    residuos = y_test - y_test_pred
    
    # Gráfico de Resíduos
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Gráfico de dispersão de resíduos
    sns.scatterplot(x=y_test_pred, y=residuos, ax=axs[0])
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_title('Gráfico de Resíduos')
    axs[0].set_xlabel('Valores Previsto')
    axs[0].set_ylabel('Resíduos')
    
    # Histograma dos resíduos
    sns.histplot(residuos, bins=30, kde=True, ax=axs[1])
    axs[1].set_title('Distribuição dos Resíduos')
    axs[1].set_xlabel('Resíduos')
    axs[1].set_ylabel('Frequência')
    
    plt.tight_layout()
    
    # Exibir os gráficos no Streamlit
    st.pyplot(fig)

    # Coeficientes da regressão Lasso
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    coef = coef.sort_values(ascending=False)  # Ordenar os coeficientes por importância
    
    # Selecionando as 5 variáveis mais influentes (valores absolutos)
    top_5_coef = coef.abs().sort_values(ascending=False).head(5)
    
    # Plotando as 5 variáveis mais influentes
    fig_top5, ax_top5 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_5_coef.values, y=top_5_coef.index, ax=ax_top5, palette='viridis')
    ax_top5.set_title('Top 5 Variáveis mais Influentes')
    ax_top5.set_xlabel('Coeficiente Absoluto')
    ax_top5.set_ylabel('Variáveis')
    
    # Exibindo o gráfico no Streamlit
    st.pyplot(fig_top5)

if __name__ == '__main__':
    run()