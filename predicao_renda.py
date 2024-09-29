import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


    


def run():
    # Configurações iniciais de exibição
    st.title("Regressão Lasso - Avaliação de Desempenho")
   
    df_final = pd.read_csv(r"variavel_x.xlsx")
    dummies = pd.read_csv(r"variavel_y.xlsx")

    
    # Separando recursos e alvo
    X = df_final  # Features
    y = dummies['renda_log']  # Target
    
    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinamento do modelo Lasso
    model_lasso = Lasso(alpha=0.01, max_iter=1000)  # Reduzir iterações para melhorar performance
    model_lasso.fit(X_train, y_train)
    
    # Previsões
    y_train_pred = model_lasso.predict(X_train)
    y_test_pred = model_lasso.predict(X_test)
    
    # Avaliação do desempenho
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Exibir métricas no Streamlit
    st.write(f'Treinamento MSE: {train_mse:.2f}, R²: {train_r2:.2f}')
    st.write(f'Teste MSE: {test_mse:.2f}, R²: {test_r2:.2f}')
    
    # Verificação de Overfitting
    if train_r2 - test_r2 > 0.1:
        st.warning("Possível overfitting detectado.")
    
    # Análise de Resíduos
    residuos = y_test - y_test_pred
    
    # Gráfico de Resíduos e Histograma
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Gráfico de dispersão dos resíduos
    sns.scatterplot(x=y_test_pred, y=residuos, ax=axs[0])
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_title('Gráfico de Resíduos')
    axs[0].set_xlabel('Valores Previstos')
    axs[0].set_ylabel('Resíduos')
    
    # Histograma dos resíduos
    sns.histplot(residuos, bins=30, kde=True, ax=axs[1])
    axs[1].set_title('Distribuição dos Resíduos')
    axs[1].set_xlabel('Resíduos')
    axs[1].set_ylabel('Frequência')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Coeficientes da regressão Lasso
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    coef = coef.sort_values(ascending=False)  # Ordenar os coeficientes por importância
    
    # Selecionando as 5 variáveis mais influentes (valores absolutos)
    top_5_coef = coef.abs().sort_values(ascending=False).head(5)
    
    # Plotando as 5 variáveis mais influentes
    fig_top5, ax_top5 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top_5_coef.values, y=top_5_coef.index, ax=ax_top5, palette='viridis')
    ax_top5.set_title('Top 5 Variáveis mais Influentes')
    ax_top5.set_xlabel('Coeficiente Absoluto')
    ax_top5.set_ylabel('Variáveis')
    
    st.pyplot(fig_top5)

if __name__ == '__main__':
    run()
