import streamlit as st
from PIL import Image

def run():
    # Título da subpágina
    st.title("Regressão Linear - Avaliação de Desempenho")
    
    # Parâmetros de desempenho
    st.write("Treinamento MSE: 53643120.25768659, R²: 0.22699852021959377")
    st.write("Teste MSE: 51920808.33702342, R²: 0.21137142575900558")
    
    # Mostrar a figura do gráfico de resíduos
    st.image("grafico_residuos.png", caption='Gráfico de Resíduos', use_column_width=True)
    
    # Mostrar a figura do gráfico dos 5 coeficientes
    st.image("grafico_top5_coeficientes.png", caption='Gráfico Top 5 Coeficientes', use_column_width=True)

# Execute a função run quando o script for chamado
if __name__ == "__main__":
    run()
