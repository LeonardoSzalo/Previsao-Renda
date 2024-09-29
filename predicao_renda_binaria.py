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
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
import pandas as pd

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
    st.title("Modelo de Classificação com XGBoost")

    # Configurações iniciais
    color_palette = ['#023047', '#e85d04', '#0077b6', '#0096c7', '#ff9c33']
    sns.set_palette(sns.color_palette(color_palette))
    pd.set_option('display.max_columns', None)

    # Carregar o seu DataFrame renda
    renda_1 = pd.read_csv(r"previsao_de_renda.xls")


    def renda_acima_media(renda):
        if renda < 3500:
            return '0'
        else:
            return '1'
    renda_1['renda_acima_media'] = renda_1['renda'].apply(renda_acima_media)   

    renda_1.dropna(inplace=True)

    renda_1.drop(['Unnamed: 0', 'data_ref', 'id_cliente'], axis=1, inplace=True)

    X = renda_1.drop(['posse_de_imovel', 'qtd_filhos', 'tipo_residencia', 'qt_pessoas_residencia', 'renda', 'renda_acima_media'], axis=1)

    y=renda_1.renda_acima_media

    X=pd.get_dummies(X, drop_first=True)

    y = y.astype(int)

      # Dividindo os dados em treino e teste
  # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

    # Inicializando o modelo XGBoost
    model = XGBClassifier(eval_metric='logloss')

    # Treinando o modelo
    model.fit(X_train, y_train)

    # Fazendo previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Avaliação do modelo
    acuracia = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Exibir resultados no Streamlit
    st.subheader("Resultados do Modelo XGBoost")
    st.write(f"""Esta solução alternativa divide a variável resposta em 'acima da mediana' ou 'abaixo da mediana' 
    para fazer as predições""")
    st.write(f"Acurácia: {acuracia:.2f}")
    st.write(f"AUC ROC: {roc_auc:.2f}")

    # Exibir o relatório de classificação diretamente
    st.write("Relatório de Classificação:")
    report = classification_report(y_test, y_pred)
    st.text(report)

    # Plotando a curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'XGBoost (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax.set_title('Curva ROC')
    ax.legend()

    # Exibir o gráfico da curva ROC no Streamlit
    st.subheader("Curva ROC")
    st.pyplot(fig)

# Iniciar o Streamlit
if __name__ == '__main__':
    run()
