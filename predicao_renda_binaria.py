import streamlit as st
from PIL import Image
import pandas as pd

def run():
    # Título da subpágina
    st.title("Modelo de Classificação com XGBoost")

    # Resultados do modelo
    accuracy = 0.7371949584338965
    classification_report = (
        "              precision    recall  f1-score   support\n\n"
        "           0       0.71      0.75      0.73      1764\n"
        "           1       0.77      0.72      0.74      1965\n\n"
        "    accuracy                           0.74      3729\n"
        "   macro avg       0.74      0.74      0.74      3729\n"
        "weighted avg       0.74      0.74      0.74      3729\n"
    )
    roc_auc = 0.8151432668062984



    # Exibir a figura
    st.image("previsao_alternativa.png", caption="Curva ROC", use_column_width=True)

# Chamar a função run() em um contexto apropriado, como em um if __name__ == "__main__":
if __name__ == "__main__":
    run()
