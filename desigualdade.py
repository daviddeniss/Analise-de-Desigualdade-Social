import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

# Configurações
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.float_format', '{:.2f}'.format)

# Constantes
COUNTRIES = {
    'BR': 'Brazil', 'AR': 'Argentina', 'CL': 'Chile',
    'CO': 'Colombia', 'PE': 'Peru', 'UY': 'Uruguay',
    'PY': 'Paraguay', 'EC': 'Ecuador', 'BO': 'Bolivia',
    'VE': 'Venezuela', 'GY': 'Guyana', 'SR': 'Suriname'
}

INDICATORS = {
    'SI.POV.GINI': 'Gini',
    'NY.GDP.PCAP.PP.KD': 'PIB_per_capita',
    'SI.DST.10TH.10': 'Renda_top_10%',
    'SI.DST.02ND.20': 'Renda_top_5%'
}

BASE_URL = "https://api.worldbank.org/v2/country/{}/indicator/{}?format=json&date=2010:2022&per_page=100"

@lru_cache(maxsize=32)
def fetch_wb_indicator(country_code, indicator_id):
    """Busca um indicador específico para um país com cache"""
    try:
        url = BASE_URL.format(country_code, indicator_id)
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                return [(item['date'], item['value']) for item in data[1] if item['value'] is not None]
        return []
    except Exception:
        return []

def fetch_all_data_parallel():
    """Extrai todos os dados em paralelo para melhor performance"""
    start_time = time.time()
    all_data = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for country_code in COUNTRIES.keys():
            for indicator_id in INDICATORS.keys():
                futures.append(executor.submit(
                    fetch_wb_indicator, country_code, indicator_id
                ))
        
        for future in futures:
            result = future.result()
            if result:
                all_data.extend(result)
    
    print(f"⏱️ Dados extraídos em {time.time() - start_time:.2f} segundos")
    return all_data

def process_data(raw_data):
    """Processa os dados brutos em um DataFrame estruturado"""
    df = pd.DataFrame(raw_data, columns=['Year', 'Value', 'Country', 'Indicator'])
    
    # Transformação para formato wide
    df_wide = df.pivot_table(
        index=['Country', 'Year'],
        columns='Indicator',
        values='Value'
    ).reset_index()
    
    # Conversão de tipos e limpeza
    df_wide['Year'] = df_wide['Year'].astype(int)
    for col in INDICATORS.values():
        if col in df_wide.columns:
            df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')
    
    return df_wide.dropna(subset=['Gini', 'PIB_per_capita'], how='all')

def plot_evolution(df):
    """Evolução temporal do coeficiente de Gini"""
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x='Year',
        y='Gini',
        hue='Country',
        style='Country',
        markers=True,
        dashes=False,
        linewidth=2
    )
    plt.title('Evolução da Desigualdade na América do Sul (2010-2022)')
    plt.ylabel('Coeficiente de Gini')
    plt.xlabel('Ano')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_pib_vs_inequality(df):
    """Relação entre PIB per capita e desigualdade"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='PIB_per_capita',
        y='Gini',
        hue='Country',
        size='Renda_top_10%',
        sizes=(50, 300),
        alpha=0.7
    )
    plt.title('Relação entre PIB per capita e Desigualdade')
    plt.xlabel('PIB per capita (PPP, USD)')
    plt.ylabel('Coeficiente de Gini')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_income_concentration(df):
    """Concentração de renda nos países mais desiguais"""
    top_countries = df.groupby('Country')['Gini'].mean().nlargest(5).index
    top_df = df[df['Country'].isin(top_countries)].copy()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=top_df,
        x='Country',
        y='Renda_top_10%',
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black"
        }
    )
    plt.title('Participação do Top 10% na Renda (Países mais Desiguais)')
    plt.ylabel('% da Renda Nacional')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xticks(rotation=45)
    plt.show()

def plot_correlation_matrix(df):
    """Matriz de correlação entre indicadores"""
    numeric_cols = ['Gini', 'PIB_per_capita', 'Renda_top_10%', 'Renda_top_5%']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Coeficiente de Correlação'}
    )
    plt.title('Correlação entre Indicadores Socioeconômicos')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    """Pipeline principal"""
    print("⏳ Iniciando extração de dados...")
    
    # Extração
    raw_data = []
    for country_code, country_name in COUNTRIES.items():
        for indicator_id, indicator_name in INDICATORS.items():
            data = fetch_wb_indicator(country_code, indicator_id)
            for year, value in data:
                raw_data.append({
                    'Year': year,
                    'Value': value,
                    'Country': country_name,
                    'Indicator': indicator_name
                })
    
    if not raw_data:
        print("⚠️ Falha na extração de dados. Verifique sua conexão com a internet.")
        return
    
    # Processamento
    df = process_data(raw_data)
    
    if df.empty:
        print("⚠️ Dados insuficientes após processamento.")
        return
    
    # Análise
    print("\n🔍 Estatísticas Descritivas:")
    print(df.describe())
    
    print("\n🌎 Países com dados disponíveis:")
    print(df['Country'].unique())
    
    # Visualizações
    plot_evolution(df)
    plot_pib_vs_inequality(df)
    plot_income_concentration(df)
    plot_correlation_matrix(df)
    
    # Salvando
    df.to_csv('dados_desigualdade_sul_americana.csv', index=False)
    print("\n💾 Dados salvos em 'dados_desigualdade_sul_americana.csv'")

if __name__ == "__main__":
    main()