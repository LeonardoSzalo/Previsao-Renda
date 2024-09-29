import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st

def run():
    # Configurações iniciais
    color_palette = ['#023047', '#e85d04', '#0077b6', '#0096c7', '#ff9c33']
    sns.set_palette(sns.color_palette(color_palette))
    pd.set_option('display.max_columns', None)

    # Carregamento dos dados
    https://github.com/LeonardoSzalo/Previsao-Renda/blob/master/previsao_de_renda.xls
    renda = pd.read_csv(r"https://raw.githubusercontent.com/LeonardoSzalo/Previsao-Renda/refs/heads/master/previsao_de_renda.xls")
    renda['data_ref'] = pd.to_datetime(renda['data_ref'], format='%Y-%m-%d')

    # Função para criar os gráficos
    def create_plot(data, x_col, y_col=None, hue_col=None, plot_type='hist', title='', xlabel='', ylabel='', kde=False, bins=30, ylim=None):
        # Criar a figura
        plt.figure(figsize=(10, 8))

        if plot_type == 'hist':
            sns.histplot(data=data[data[x_col] < ylim], x=x_col, hue=hue_col, kde=kde, bins=bins)
        elif plot_type == 'scatter':
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col)
        elif plot_type == 'box':
            sns.boxplot(data=data, x=x_col, y=y_col, hue=hue_col)
        elif plot_type == 'count':
            sns.countplot(data=data, x=x_col, hue=hue_col)
        elif plot_type == 'reg':
            sns.regplot(data=data, x=x_col, y=y_col, line_kws={'color': 'red'})

        # Títulos e rótulos
        plt.title(title, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Ajustar layout
        plt.tight_layout()

        # Salvar a figura em um buffer para exibir no Streamlit
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

    # Definir os gráficos disponíveis e seus textos descritivos
    graphs = {
        'Renda até 25k (Histograma)': {
            'plot_func': lambda: create_plot(renda, x_col='renda', hue_col='sexo', plot_type='hist', title='Distribuição de Renda até 25k', xlabel='Renda', ylabel='Frequência', kde=True, bins=50, ylim=25000),
            'description': 'Distribuição da renda dos participantes, com um limite de até 25k, categorizada por sexo.'
        },
        'Renda por Tempo de Emprego (Regplot)': {
            'plot_func': lambda: create_plot(renda, x_col='tempo_emprego', y_col='renda', plot_type='reg', title='Renda por Tempo de Emprego (Regplot)', xlabel='Tempo de Emprego', ylabel='Renda'),
            'description': 'Este gráfico mostra a relação entre o tempo de emprego e a renda dos participantes, com uma linha de tendência em vermelho.'
        },
        'Renda por Posse de Veículo (Boxplot)': {
            'plot_func': lambda: create_plot(renda, x_col='posse_de_veiculo', y_col='renda', hue_col='sexo', plot_type='box', title='Renda por Posse de Veículo', xlabel='Posse de Veículo', ylabel='Renda'),
            'description': 'Comparação da renda dos participantes com base na posse de veículos, separada por sexo.'
        },
        'Distribuição por Sexo (Countplot)': {
            'plot_func': lambda: create_plot(renda, x_col='sexo', plot_type='count', title='Distribuição por Sexo', xlabel='Sexo', ylabel='Contagem'),
            'description': 'Contagem da distribuição dos participantes por sexo.'
        }
    }

    # Interface no Streamlit
    st.title("Visualização de Dados de Renda")
    
    # Cria um menu suspenso para selecionar o gráfico
    selected_graph = st.selectbox('Selecione um gráfico para exibir:', list(graphs.keys()))
    
    # Exibe o gráfico selecionado
    if selected_graph:
        buf = graphs[selected_graph]['plot_func']()
        st.image(buf)
        
        # Exibe a descrição
        st.write(graphs[selected_graph]['description'])


if __name__ == "__main__":
    run()