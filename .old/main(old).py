# app.py

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu
import altair as alt
from datetime import timedelta
from streamlit_option_menu import option_menu

# --- Load data and model ---
@st.cache_data
def load_data():
    df = pd.read_csv('./data/brent.csv', parse_dates=['date'])
    df = df.set_index('date').asfreq('D')
    df['value'] = df['value'].interpolate()
    return df

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

# --- Feature engineering ---
def create_features(df):
    df_feat = df.copy()
    df_feat['lag_1'] = df_feat['value'].shift(1)
    df_feat['lag_7'] = df_feat['value'].shift(7)
    df_feat['lag_30'] = df_feat['value'].shift(30)
    df_feat['rolling_mean_7'] = df_feat['value'].rolling(7).mean()
    df_feat['rolling_mean_30'] = df_feat['value'].rolling(30).mean()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['trend'] = np.arange(len(df_feat))
    
    
    return df_feat.dropna()

# --- Forecasting logic ---
def forecast(df, model, n_days):
    last_df = df.copy()
    for _ in range(n_days):
        features = create_features(last_df)[-1:]
        X_pred = features.drop(columns=['value'])
        y_pred = model.predict(X_pred)[0]
        next_date = last_df.index[-1] + timedelta(days=1)
        next_row = pd.DataFrame({'value': [y_pred]}, index=[next_date])
        last_df = pd.concat([last_df, next_row])
    forecast_part = last_df.tail(n_days)
    forecast_part = forecast_part[['value']].rename(columns={'value': 'forecast'})
    return forecast_part

# --- Streamlit UI ---

st.set_page_config(page_title="Brent Forecast", layout="wide")


with st.sidebar:
    pages = {
        "Previsão": "forecast",
        "Histórico do brent": "history",
        "Análise das séries ": "analysis",
        "Sobre": "about"
    }

    selected = option_menu(
        menu_title="Menu",
        options=list(pages.keys()),
        icons=["graph-up", "clock-history", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#f63366", "color": "white"},
        }
    )

    current_page = pages[selected]

# --- Page navigation logic ---
df = load_data()
model = load_model()

if current_page == "forecast":

    st.title("Previsão para o preço do Brent")

    df = load_data()
    model = load_model()

    n_days = st.slider("Selecione a quantidade de dias para previsão", min_value=1, max_value=30, value=7)

    forecast_df = forecast(df, model, n_days)

    st.subheader(f"Forecast dos próximos {n_days} dias",)
    
    st.line_chart(forecast_df)

    st.markdown("<h3 style='text-align: center;'>Valores dos últimos 5 dias</h3>", unsafe_allow_html=True)
    df_toview = df.iloc[-5:,slice(None)].copy(deep=True)
    df_toview['value'] = df_toview['value'].round(2)
    df_toview.index = df_toview.index.strftime('%d-%m-%Y')
    df_toview.index.name = 'Data'   
    df_toview.rename(columns={'value': 'US$/barril'}, inplace=True)
    st.dataframe(df_toview[['US$/barril']].tail(5),)


    st.subheader("Últimos 90 dias + previsão realizada")
    # Combine last 90 days of history with forecast, labeling each part
    history = df[['value']].iloc[-90:].copy()
    history['tipo'] = 'Histórico'
    forecast_plot = forecast_df.copy()
    forecast_plot = forecast_plot.rename(columns={'forecast': 'value'})
    forecast_plot['tipo'] = 'Previsão'
    combined = pd.concat([history, forecast_plot])
    combined.reset_index(inplace=True)
    combined = combined.rename(columns={'index': 'Data'})

    # Plot with color distinction
    chart = alt.Chart(combined).mark_line().encode(
        x='Data:T',
        y=alt.Y('value:Q', axis=alt.Axis(format=".2f"),title=None),
        color=alt.Color('tipo:N', scale=alt.Scale(domain=['Histórico', 'Previsão'], range=['#1f77b4', '#f63366'])),
        tooltip=[alt.Tooltip('Data:T'), alt.Tooltip('value:Q', format=".2f"),alt.Tooltip('tipo:N')]
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    st.download_button("Download Forecast", forecast_df.reset_index().to_csv(index=True), file_name="brent_forecast.csv")


elif current_page == "history":

    st.title("Histórico do Brent")
    st.markdown("### Principais pontos de virada do preço do Brent (2002-2022)")

    # Data for the table
    table_data = [
        ["24-06-2002", "Vale", "$24,44", 
         "Início da guerra do Iraque / baixa demanda"],
        ["18-11-2002", "Vale", "$24,31", 
         "Período semelhante, ainda pré-guerra do Iraque"],
        ["28-04-2003", "Vale", "$24,43", 
         "Fim dos principais combates no Iraque"],
        ["08-07-2008", "Pico", "$136,51", 
         "Grande alta antes da crise de 2008"],
        ["14-04-2011", "Pico", "$121,80", 
         "Primavera Árabe / colapso do petróleo na Líbia"],
        ["16-03-2012", "Pico", "$125,33", 
         "Sanções ao Irã e tensões geopolíticas"],
        ["14-04-2020", "Vale", "$19,29", 
         "Queda de demanda por COVID"],
        ["14-06-2022", "Pico", "$121,83", 
         "Guerra na Ucrânia, oferta restrita"],
    ]

    columns = ["Data", "Tipo", "Preço (US$)", "Análises realizadas"]

    # Convert to DataFrame for Streamlit display
    table_df = pd.DataFrame(table_data, columns=columns )
    # Color peaks and troughs
    def highlight_row(row):
        color = '#2ecc71' if row['Tipo'] == 'Pico' else '#e74c3c'
        return [f'background-color: {color}; color: #fff'] * len(row)
    
    st.dataframe(table_df.style.apply(highlight_row, axis=1),hide_index=True, 
                 use_container_width=True)

    st.markdown("#### Evolução histórica do preço do Brent")
    
    
    
    fig = px.line(df[['value']])
    fig.update_layout(
        showlegend=False,
        yaxis_title="US$",
        xaxis_title="Evolução no tempo"
    )
    st.plotly_chart(fig)
    # st.line_chart(df['value'])
    st.markdown("<h3 style='text-align: center;'>Valores dos últimos 5 dias</h3>", unsafe_allow_html=True)

    
    last5 = df[['value']].tail(5).copy()
    last5.index = last5.index.strftime('%d-%m-%Y')
    last5.rename(columns={'value': 'US$/barril'}, inplace=True)
    # Ajusta o tamanho das colunas usando st.table (renderiza como tabela fixa e centralizada)
    st.table(last5.style.format({'US$/barril': '{:.2f}'}))
    # st.write_stream( df.iloc[-5:])
    
    st.write('''Podemos observar que o petróleo no longo prazo possui um
             comportamento de alta variação, com vários pontos de virada e está
             altamente correlacionado a demanda global, com grande impacto em 
             políticas locais e globais.
                ''')

elif current_page == "analysis":
    ...

elif current_page == "about":
    st.title("Sobre")
    st.header("Tech Challenge - FIAP - Módulo IV")
    st.html("")
    st.html('''
            <p> Criado por: Vinicius de Souza Machado | RM: 358294 
            <p> <i>Este aplicativo foi desenvolvido para previsão do preço do 
            Brent utilizando aprendizado de máquina. </i> <p>
            ''')
    
    
    
    


