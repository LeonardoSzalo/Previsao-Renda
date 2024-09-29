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

st.set_page_config(page_title="Previsao de Renda",layout="wide", page_icon = 'https://www.cdc.gov/nchs/images/nhanes/NHANES-Trademark.png?_=04691')


# Dicionário para armazenar as diferentes páginas
pages = {
    "Sobre o projeto": intro,
    "Análise demográfica": demografico_streamlit,
    "Previsão de renda": predicao_renda,
    "Solução alternativa": predicao_renda_binaria
    
}

# Sidebar para selecionar a página
st.sidebar.title("Menu")
selection = st.sidebar.radio("Ir para", list(pages.keys()))

# Carregar a página correspondente
page = pages[selection]
page.run()  # Executa a função `run()` de cada script
