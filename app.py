import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplcyberpunk
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

# Criação da barra lateral
sidebar = st.sidebar

# Adicionar logo à barra lateral
logo = 'logo.png'  # Substitua pelo caminho correto para o seu logo
sidebar.image(logo, use_container_width=True)

# Widget de upload de arquivo na barra lateral
uploaded_file = sidebar.file_uploader('Use um arquivo DPT', type="dpt")

# Inicializar a variável de estado para exibir o botão e a mensagem
if "show_button" not in st.session_state:
    st.session_state.show_button = True

# Verifica se um arquivo foi carregado
if uploaded_file is not None:
    # Ler o conteúdo do arquivo em um DataFrame
    dataframe = pd.read_csv(uploaded_file, sep = '\t', header = None, index_col = 0,
                            skiprows = 0, decimal = '.', names = ['Número de Onda', 'Transmitância'])

    # Filtrar a faixa de 1800 a 900
    dados_coletados = dataframe.loc[4000:400]

    # Exibir as primeiras cinco linhas do DataFrame na barra lateral
    sidebar.write('Arquivo Carregado!')
    sidebar.dataframe(dados_coletados.head(5))

    # Criar colunas para o gráfico e resultados
    col1 = st.columns(1)[0]  # Pegando a primeira (e única) coluna

    # Exibir gráfico FTIR antes de pressionar o botão
    if st.session_state.show_button:
        with col1:
            fig = plt.figure(figsize=(13, 6))
            plt.style.use("cyberpunk")

            # Criar sua linha
            plt.plot(dados_coletados, lw=2, color='green')  # Linha na cor verde

            # Adicionar efeitos de brilho
            mplcyberpunk.add_glow_effects()

            # Formatação do gráfico
            plt.gca().invert_xaxis()
            plt.title('Espectro FTIR', pad=10, fontsize=30, fontname='Cambria')
            plt.xlabel('Número de Onda ($\mathregular{cm^-¹}$)', labelpad=17, fontsize=26, fontname='Cambria')
            plt.ylabel('Transmitância Normalizada', labelpad=15, fontsize=28, fontname='Cambria')
            plt.xticks(np.arange(400, 4000 + 100, 100), fontsize=18, fontname='Cambria')
            plt.gca().tick_params(axis='x', pad=20)  # Ajusta o espaço entre os rótulos e a linha
            plt.xlim(4000, 400)
            # plt.ylim(dados_coletados.min().min() - 0.5, 100.5)
            plt.yticks(fontsize=18, fontname='Cambria')
            plt.gca().tick_params(axis='y', pad=20)  # Ajusta o espaço entre os rótulos e a linha
            st.pyplot(fig)

    # Exibir a mensagem e o botão "Continuar" apenas se for permitido
    st.info('Espectro medido corretamente! Clique em "continuar"')

    if st.button('Continuar'):
        st.session_state.show_button = False  # Ocultar mensagem e botão após o clique

    # Exibir o gráfico de pizza com as probabilidades após o botão ser pressionado
    if not st.session_state.show_button:
        
        # Carregar os modelos treinados
        with open('bin1.pkl', 'rb') as f:
            model1 = pickle.load(f)
            
        with open('bin2.pkl', 'rb') as f:
            model2 = pickle.load(f)

        # Pré-tratamento (SNV)
        dados_intervalo = dados_coletados.loc[4000:400] ## SELECIONAR INTERVALO para SNV
        dados_tratados = (dados_int - dados_int.mean(axis=0)) / dados_int.std(axis=0)

        # Matriz Transposta (n_amostras, n_variáveis)
        X = np.array(np.transpose(dados_tratados)) 

        # 1º Classificador: bin1 (Alto-Médio vs Baixo)
        pred_bin1 = model_bin1.predict(X)[0]
        prob_bin1 = model_bin1.predict_proba(X)[0]
        
        if pred_bin1 == 'Baixo':
            st.success(f'A amostra foi classificada como: **Baixo**')
            prob_baixo = prob_bin1[model_bin1.classes_ == 'Baixo'][0] * 100
            prob_altomedio = prob_bin1[model_bin1.classes_ == 'Alto-Médio'][0] * 100
        
            # Gráfico de pizza
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie([prob_baixo, prob_altomedio], labels=['Baixo', 'Alto-Médio'],
                   autopct='%1.2f%%', startangle=90, colors=['green', 'gray'])
            ax.set_title('Probabilidades - Nível 1', fontsize=10)
            st.pyplot(fig)
        
        else:
            st.info(f'A amostra foi classificada como: **Alto-Médio**')
        
            # 2º Classificador: bin2 (Alto vs Médio)
            pred_bin2 = model_bin2.predict(X)[0]
            prob_bin2 = model_bin2.predict_proba(X)[0]
        
            prob_alto = prob_bin2[model_bin2.classes_ == 'Alto'][0] * 100
            prob_medio = prob_bin2[model_bin2.classes_ == 'Médio'][0] * 100
        
            st.success(f'Detalhamento: **{pred_bin2}**')
        
            # Gráfico de pizza
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie([prob_alto, prob_medio], labels=['Alto', 'Médio'],
                   autopct='%1.2f%%', startangle=90, colors=['red', 'orange'])
            ax.set_title('Probabilidades - Nível 2', fontsize=10)
            st.pyplot(fig)

else:
    st.markdown('''<h1 style="color: orange; font-size: 35px;">Diagnóstico de Brucelose Bovina</h1>''', unsafe_allow_html=True)
    # Subtítulo (h3)
    st.markdown('''<h3 style="color: white; font-size: 20px;">Carregue um espectro FTIR para análise</h3>''', unsafe_allow_html=True)





