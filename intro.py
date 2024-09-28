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

import pandas as pd
import streamlit as st

def run():
    # Função para criar um cabeçalho centralizado no Streamlit usando Markdown
    def centered_header(title):
        st.markdown(f"<h2 style='text-align: center; font-family: Arial, sans-serif;'>{title}</h2>", unsafe_allow_html=True)
    
    # Função para criar texto estilizado usando Markdown
    def styled_text(text):
        return f"<p style='font-family: Arial, sans-serif;'>{text}</p>"

    # Título da página
    centered_header("Sobre o projeto")
    
    # Selecionar seções progressivamente
    show_section = st.selectbox("Select Section", ["Introdução", "Variáveis", "Mais informações"])
    
    if show_section == "Introdução":
        st.markdown(styled_text(
            """ 
            Este é um problema de ciência de dados desenvolvido como parte do processo de aprendizado no curso da EBAC. 
            O conjunto de dados utilizado foi modificado e fornecido pela EBAC (https://ebaconline.com.br/).

            A predição de renda é um problema de ciência de dados que envolve estimar o nível de renda de um indivíduo ou grupo com base 
            em características demográficas, sociais e econômicas. Esse tipo de análise é crucial para setores como marketing, crédito, 
            e políticas públicas, pois ajuda na tomada de decisões informadas, personalização de serviços e alocação de recursos.

            Essas previsões são essenciais para identificar padrões socioeconômicos, direcionar campanhas de marketing de forma mais eficaz, 
            avaliar riscos de crédito e entender a distribuição de riqueza em uma população.
        
            """
        ), unsafe_allow_html=True)
        
    elif show_section == "Variáveis":
        # Definindo os dados da tabela
        data = {
            "Variável": [
                "data_ref",
                "id_cliente",
                "sexo",
                "posse_de_veiculo",
                "posse_de_imovel",
                "qtd_filhos",
                "tipo_renda",
                "educacao",
                "estado_civil",
                "tipo_residencia",
                "idade",
                "tempo_emprego",
                "qt_pessoas_residencia",
                "renda",
            ],
            "Descrição": [
                "Data da coleta dos dados",
                "Identificação do cliente",
                "M = 'Masculino'; F = 'Feminino'",
                "T = 'possui'; F = 'não possui'",
                "T = 'possui'; F = 'não possui'",
                "Quantidade de filhos",
                "Tipo de renda (ex: assalariado, autônomo etc)",
                "Nível de escolaridade",
                "Estado Civil",
                "Tipo de residência (casa, apartamento etc)",
                "Idade",
                "Tempo de emprego em anos",
                "Quantidade de pessoas na residência",
                "Renda",
            ],
            "Tipo": [
                "Object",
                "Int",
                "Object",
                "Bool",
                "Bool",
                "Int",
                "Object",
                "Object",
                "Object",
                "Object",
                "Int",
                "Float",
                "Float",
                "Float",
            ],
        }

        # Criando um DataFrame a partir dos dados
        df_variables = pd.DataFrame(data)

        # Exibindo a tabela no Streamlit
        st.subheader("Descrição das Variáveis")
        st.table(df_variables)
    
    
    elif show_section == "Mais informações":
       

        st.markdown(
        """
        To see the complete work and analyses step by step, access my GitHub repository (in process): 
        [GitHub Repository](https://github.com/LeonardoSzalo)

        For updates and more information, connect with me on LinkedIn: 
        [LinkedIn Profile](https://www.linkedin.com/in/leonardo-szalo-11a3aa156)
        """,
        unsafe_allow_html=True  # Permite a inclusão de HTML e Markdown
        )
