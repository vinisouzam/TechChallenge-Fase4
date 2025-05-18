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

# --- Feature engineering (versão alinhada com modelo final) ---
def create_features(df):
    df_feat = df.copy()
    for lag in [1, 3, 7, 14, 30]:
        df_feat[f'lag_{lag}'] = df_feat['value'].shift(lag)
    for window in [3, 7, 14, 30]:
        df_feat[f'rolling_mean_{window}'] = df_feat['value'].rolling(window).mean()
    df_feat['rolling_std_7'] = df_feat['value'].rolling(7).std()
    df_feat['exp_moving_avg_7'] = df_feat['value'].ewm(span=7).mean()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['is_month_start'] = df_feat.index.is_month_start.astype(int)
    df_feat['is_month_end'] = df_feat.index.is_month_end.astype(int)
    df_feat['trend'] = np.arange(len(df_feat))
    df_feat = df_feat.dropna()
    return df_feat

# --- Forecasting logic com previsão recursiva ---
def recursive_forecast(model, df, steps=30):
    df_feat = create_features(df)
    current = df_feat.copy()
    forecast = []
    for _ in range(steps):
        last_row = current.iloc[-1:]
        next_date = last_row.index[0] + timedelta(days=1)
        new_row = pd.DataFrame(index=[next_date])

        for lag in [1, 3, 7, 14, 30]:
            new_row[f'lag_{lag}'] = current['value'].iloc[-lag]
        for window in [3, 7, 14, 30]:
            new_row[f'rolling_mean_{window}'] = current['value'].iloc[-window:].mean()
        new_row['rolling_std_7'] = current['value'].iloc[-7:].std()
        new_row['exp_moving_avg_7'] = current['value'].ewm(span=7).mean().iloc[-1]
        new_row['dayofweek'] = next_date.dayofweek
        new_row['month'] = next_date.month
        new_row['quarter'] = (next_date.month - 1) // 3 + 1
        new_row['dayofyear'] = next_date.dayofyear
        new_row['is_month_start'] = int(next_date.is_month_start)
        new_row['is_month_end'] = int(next_date.is_month_end)
        new_row['trend'] = current['trend'].iloc[-1] + 1

        y_pred = model.predict(new_row)[0]
        new_row['value'] = y_pred
        current = pd.concat([current, new_row])
        forecast.append((next_date, y_pred))

    forecast_df = pd.DataFrame(forecast, columns=['date', 'forecast']).set_index('date')
    return forecast_df

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

# --- Page logic ---
df = load_data()
model = load_model()

if current_page == "forecast":
    st.title("Previsão para o preço do Brent")
    n_days = st.slider("Selecione a quantidade de dias para previsão", min_value=1, max_value=30, value=7)
    forecast_df = recursive_forecast(model, df, steps=n_days)

    st.subheader(f"Forecast dos próximos {n_days} dias")
    st.line_chart(forecast_df)

    st.markdown("<h3 style='text-align: center;'>Valores dos últimos 5 dias</h3>", unsafe_allow_html=True)
    df_toview = df.iloc[-5:,slice(None)].copy()
    df_toview['value'] = df_toview['value'].round(2)
    df_toview.index = df_toview.index.strftime('%d-%m-%Y')
    df_toview.index.name = 'Data'
    df_toview.rename(columns={'value': 'US$/barril'}, inplace=True)
    st.dataframe(df_toview[['US$/barril']].tail(5))

    st.subheader("Últimos 90 dias + previsão realizada")
    history = df[['value']].iloc[-90:].copy()
    history['tipo'] = 'Histórico'
    forecast_plot = forecast_df.rename(columns={'forecast': 'value'})
    forecast_plot['tipo'] = 'Previsão'
    combined = pd.concat([history, forecast_plot])
    combined.reset_index(inplace=True)
    combined = combined.rename(columns={'index': 'Data'})

    chart = alt.Chart(combined).mark_line().encode(
        x='Data:T',
        y=alt.Y('value:Q', axis=alt.Axis(format=".2f"), title=None),
        color=alt.Color('tipo:N', scale=alt.Scale(domain=['Histórico', 'Previsão'], range=['#1f77b4', '#f63366'])),
        tooltip=[alt.Tooltip('Data:T'), alt.Tooltip('value:Q', format=".2f"), alt.Tooltip('tipo:N')]
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    st.download_button("Download Forecast", forecast_df.reset_index().to_csv(index=True), file_name="brent_forecast.csv")

elif current_page == "history":
    st.title("Histórico do Brent")
    st.markdown("### Principais pontos de virada do preço do Brent (2002-2022)")
    table_data = [
        ["24-06-2002", "Vale", "$24,44", "Início da guerra do Iraque / baixa demanda"],
        ["18-11-2002", "Vale", "$24,31", "Período semelhante, ainda pré-guerra do Iraque"],
        ["28-04-2003", "Vale", "$24,43", "Fim dos principais combates no Iraque"],
        ["08-07-2008", "Pico", "$136,51", "Grande alta antes da crise de 2008"],
        ["14-04-2011", "Pico", "$121,80", "Primavera Árabe / colapso do petróleo na Líbia"],
        ["16-03-2012", "Pico", "$125,33", "Sanções ao Irã e tensões geopolíticas"],
        ["14-04-2020", "Vale", "$19,29", "Queda de demanda por COVID"],
        ["14-06-2022", "Pico", "$121,83", "Guerra na Ucrânia, oferta restrita"],
    ]
    columns = ["Data", "Tipo", "Preço (US$)", "Análises realizadas"]
    table_df = pd.DataFrame(table_data, columns=columns)
    def highlight_row(row):
        color = '#2ecc71' if row['Tipo'] == 'Pico' else '#e74c3c'
        return [f'background-color: {color}; color: #fff'] * len(row)
    st.dataframe(table_df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)

    st.markdown("#### Evolução histórica do preço do Brent")
    fig = px.line(df[['value']])
    fig.update_layout(showlegend=False, yaxis_title="US$", xaxis_title="Evolução no tempo")
    st.plotly_chart(fig)

    st.markdown("<h3 style='text-align: center;'>Valores dos últimos 5 dias</h3>", unsafe_allow_html=True)
    last5 = df[['value']].tail(5).copy()
    last5.index = last5.index.strftime('%d-%m-%Y')
    last5.rename(columns={'value': 'US$/barril'}, inplace=True)
    st.table(last5.style.format({'US$/barril': '{:.2f}'}))
    st.write('''Comportamento do preço do petróleo ao longo do tempo mostra forte correlação com eventos geopolíticos e demanda global.''')

elif current_page == "analysis":
    st.title("Em breve: Análise de componentes e tendências")

elif current_page == "about":
    st.title("Sobre")
    st.header("Tech Challenge - FIAP - Módulo IV")
    st.markdown('''
        <p>Criado por: Vinicius de Souza Machado | RM: 358294</p>
        <p><i>Este aplicativo foi desenvolvido para previsão do preço do Brent utilizando aprendizado de máquina.</i></p>
    ''', unsafe_allow_html=True)