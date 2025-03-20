import streamlit as st
import pandas as pd
import os
import plotly.express as px
import re

def clean_html(text):
    return re.sub(r'<.*?>', '', str(text))

@st.cache_data
def load_data(directory="./vhi_data"):
    all_dfs = []
    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            match = re.search(r'NOAA_ID(\d+)_', file)
            if not match:
                continue  # Пропустити файли з некоректними назвами
            
            region_id = int(match.group(1))
            df = pd.read_csv(file_path, header=1, names=headers, on_bad_lines='skip')
            
            if 'empty' in df.columns:
                df.drop(columns=['empty'], inplace=True)

            df['Year'] = df['Year'].astype(str).apply(clean_html)
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
            df = df[df['VHI'] != -1]
            df['area'] = region_id

            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

region_name_to_id = {
    "вінницька": 1, "волинська": 2, "дніпропетровська": 3, "донецька": 4, "житомирська": 5, 
    "закарпатська": 6, "запорізька": 7, "івано-франківська": 8, "київська": 9, "кіровоградська": 10, 
    "луганська": 11, "львівська": 12, "миколаївська": 13, "одеська": 14, "полтавська": 15, "рівненська": 16, 
    "сумська": 17, "тернопільська": 18, "харківська": 19, "херсонська": 20, "хмельницька": 21, 
    "черкаська": 22, "чернівецька": 23, "чернігівська": 24, "крим": 25, "київ": 26, "севастополь": 27
}
region_id_to_name = {v: k.title() for k, v in region_name_to_id.items()}

st.set_page_config(layout="wide", page_title="Аналіз даних")

data = load_data()

if data.empty:
    st.error("Дані не знайдено! Завантажте файли в папку ./vhi_data")
    st.stop()

if "sort_order" not in st.session_state:
    st.session_state.sort_order = None
if "week_range" not in st.session_state:
    st.session_state.week_range = (1, 52)
if "year_range" not in st.session_state:
    st.session_state.year_range = (int(data['Year'].min()), int(data['Year'].max()))
if "indicator" not in st.session_state:
    st.session_state.indicator = "VCI"
if "region" not in st.session_state:
    st.session_state.region = "Вінницька"

def reset_filters():
    st.session_state.sort_order = None
    st.session_state.week_range = (1, 52)
    st.session_state.year_range = (int(data['Year'].min()), int(data['Year'].max()))
    st.session_state.indicator = "VCI"
    st.session_state.region = "Вінницька"

st.sidebar.header("Фільтри")

indicator = st.sidebar.selectbox("Оберіть показник", ["VCI", "TCI", "VHI"], key="indicator")

sorted_regions = [region_id_to_name[i] for i in sorted(region_id_to_name.keys())]
region = st.sidebar.selectbox("Оберіть область", sorted_regions, key="region")
region_id = region_name_to_id[region.lower()]

week_range = st.sidebar.slider("Інтервал тижнів", min_value=1, max_value=52, 
                               value=st.session_state.week_range, key="week_range")

year_range = st.sidebar.slider("Інтервал років", min_value=int(data['Year'].min()), max_value=int(data['Year'].max()), 
                               value=st.session_state.year_range, key="year_range")

def update_sort_order(order):
    st.session_state.sort_order = order

sort_asc = st.sidebar.checkbox("Сортувати за зростанням", value=(st.session_state.sort_order == "asc"), 
                               on_change=update_sort_order, args=("asc",))
sort_desc = st.sidebar.checkbox("Сортувати за спаданням", value=(st.session_state.sort_order == "desc"), 
                                on_change=update_sort_order, args=("desc",))

if sort_asc and sort_desc:
    st.session_state.sort_order = None

st.sidebar.button("Скинути фільтри", on_click=reset_filters)

filtered_data = data[
    (data["area"] == region_id) &
    (data["Year"] >= year_range[0]) & (data["Year"] <= year_range[1]) &
    (data["Week"] >= week_range[0]) & (data["Week"] <= week_range[1])
]
if st.session_state.sort_order == "asc":
    filtered_data = filtered_data.sort_values(by=indicator, ascending=True)
elif st.session_state.sort_order == "desc":
    filtered_data = filtered_data.sort_values(by=indicator, ascending=False)

tab1, tab2, tab3 = st.tabs(["Таблиця", "Графік", "Порівняння"])

with tab1:
    st.subheader(f"Відфільтровані дані ({indicator}) для {region} обл.")
    st.dataframe(filtered_data)

with tab2:
    st.subheader(f"Часовий ряд {indicator} для {region}")
    fig = px.line(filtered_data, x="Week", y=indicator, color="Year", title=f"{indicator} по тижнях", markers=True)
    st.plotly_chart(fig)

with tab3:
    comparison_data = data[
        (data["Year"] >= year_range[0]) & (data["Year"] <= year_range[1]) &
        (data["Week"] >= week_range[0]) & (data["Week"] <= week_range[1])
    ].copy()

    comparison_data["Region"] = comparison_data["area"].map(region_id_to_name)
    comparison_data["Highlight"] = comparison_data["Region"].apply(
        lambda x: "Вибрана" if x.lower() == region.lower() else "Інші"
    )
    st.subheader(f"Порівняння {indicator} по областях")
    fig_comp = px.box(
        comparison_data,
        x="Region",
        y=indicator,
        color="Highlight",
        title=f"Порівняння {indicator} між областями",
        labels={"Region": "Область"},
        color_discrete_map={"Вибрана": "green", "Інші": "lightblue"}
    )
    st.plotly_chart(fig_comp)
