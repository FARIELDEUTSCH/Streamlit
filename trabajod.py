import streamlit as st
# Configuración de la app
st.set_page_config(
    page_title="Análisis de Opiniones", 
    layout="wide",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "### Herramienta de análisis de sentimientos\nVersión 2.0"
    }
)



import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import os

# Configuración inicial para optimizar descargas
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
#torch._C._jit_set_numba_helper(True)

# Descarga de recursos NLTK (con manejo de errores)
try:
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
except Exception as e:
    st.warning(f"Error al descargar recursos NLTK: {str(e)}")

# Inicialización de modelos con manejo de errores y timeout extendido
@st.cache_resource(show_spinner="Cargando modelos de análisis...")
def load_models():
    try:
        sia = SentimentIntensityAnalyzer()
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device="cpu"
            
        )
        kw_model = KeyBERT(SentenceTransformer("paraphrase-MiniLM-L6-v2"))
        return sia, summarizer, kw_model
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")
        return None, None, None

sia, summarizer, kw_model = load_models()

# Funciones mejoradas con manejo de errores
def classify_sentiment(text):
    try:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 'Neutro'
            
        score = sia.polarity_scores(text)
        compound = score['compound']
        if compound >= 0.05:
            return 'Positivo'
        elif compound <= -0.05:
            return 'Negativo'
        return 'Neutro'
    except:
        return 'Error'

def resumir_texto(texto):
    try:
        if not isinstance(texto, str) or len(texto.split()) < 30:
            return "Texto muy corto para resumir."
        max_len = min(60, int(len(texto.split()) * 0.8))  # ajusta automáticamente
        return summarizer(texto, max_length=max_len, min_length=15, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"Error al generar resumen: {str(e)}"


def extraer_temas(textos, top_n=5):
    try:
        if not textos or len(textos) == 0:
            return ["No hay texto para analizar"]
            
        texto_concatenado = " ".join([str(t) for t in textos])
        keywords = kw_model.extract_keywords(
            texto_concatenado, 
            top_n=top_n, 
            stop_words='spanish',
            keyphrase_ngram_range=(1, 2)
        )
        return [kw[0] for kw in keywords]
    except Exception as e:
        return [f"Error al extraer temas: {str(e)}"]



# Diseño mejorado
st.title("🧠 Análisis de Opiniones de Clientes")
st.markdown("""
    <style>
    .st-emotion-cache-1kyxreq {justify-content: center;}
    .st-emotion-cache-1v0mbdj {margin: 0 auto;}
    .stPlotlyChart {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# 1. Subida de archivo con validaciones mejoradas
with st.expander("📤 Subir archivo de datos", expanded=True):
    uploaded_file = st.file_uploader(
        "Sube un archivo CSV con una columna de opiniones", 
        type=["csv", "xlsx"],
        help="El archivo debe contener una columna con texto de opiniones"
    )

if uploaded_file:
    try:
        # Manejo de diferentes formatos de archivo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', sep=';', engine='python')
        else:  # Excel
            df = pd.read_excel(uploaded_file)
            
        # Validación de datos
        if df.shape[1] != 1:
            st.error("El archivo debe contener exactamente una columna con opiniones.")
        elif df.shape[0] < 10:
            st.warning("Se recomienda cargar al menos 20 opiniones para un análisis significativo.")
        else:
            st.success(f"Archivo cargado exitosamente. {len(df)} opiniones encontradas.")
            df.columns = ['opinion']
            
            # Mostrar vista previa de datos
            with st.expander("🔍 Vista previa de los datos"):
                st.dataframe(df.head(), height=150)

            # 2. Preprocesamiento mejorado
            with st.spinner("Procesando opiniones..."):
                stop_words = set(stopwords.words('spanish') + ['si', 'sí', 'no'])
                df['tokens'] = (
                    df['opinion']
                    .astype(str)
                    .str.lower()
                    .str.replace(r'[^a-záéíóúüñ ]', '', regex=True)
                    .apply(lambda x: [word for word in x.split() if word not in stop_words and len(word) > 2])
                )
                
                # Clasificación de sentimientos
                df['sentimiento'] = df['opinion'].apply(classify_sentiment)

            # 3. Visualizaciones mejoradas
            st.subheader("📊 Análisis Exploratorio")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Nube de palabras")
                all_words = [word for tokens in df['tokens'] for word in tokens]
                if all_words:
                    wc = WordCloud(
                        width=600, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate(" ".join(all_words))
                    st.image(wc.to_array(), use_container_width=True)

                else:
                    st.warning("No hay palabras suficientes para generar nube")

            with col2:
                st.markdown("#### Distribución de Sentimientos")
                sentiment_count = df['sentimiento'].value_counts()
                fig2 = px.pie(
                    sentiment_count, 
                    values=sentiment_count.values,
                    names=sentiment_count.index,
                    hole=0.3,
                    color=sentiment_count.index,
                    color_discrete_map={
                        'Positivo': '#2ecc71',
                        'Negativo': '#e74c3c',
                        'Neutro': '#3498db',
                        'Error': '#95a5a6'
                    }
                )
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)

            # 4. Análisis detallado
            st.subheader("🔍 Análisis Detallado")
            
            tab1, tab2, tab3 = st.tabs([
                "📋 Comentarios Clasificados", 
                "📈 Palabras Clave", 
                "🤖 Análisis Avanzado"
            ])
            
            with tab1:
                st.dataframe(
                    df[['opinion', 'sentimiento']].sort_values('sentimiento'),
                    height=400,
                    column_config={
                        "opinion": "Opinión",
                        "sentimiento": st.column_config.SelectboxColumn(
                            "Sentimiento",
                            options=["Positivo", "Neutro", "Negativo", "Error"]
                        )
                    },
                    hide_index=True
                )
            
            with tab2:
                top_words = Counter([word for tokens in df['tokens'] for word in tokens]).most_common(20)
                if top_words:
                    words, freqs = zip(*top_words)
                    fig = px.bar(
                        x=words, 
                        y=freqs, 
                        labels={'x': 'Palabra', 'y': 'Frecuencia'},
                        color=freqs,
                        color_continuous_scale='Bluered'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay palabras suficientes para mostrar")
            
            with tab3:
                st.markdown("### Resumen y Temas Principales")
                
                if st.button("Generar Análisis", type="primary"):
                    with st.spinner("Analizando comentarios..."):
                        try:
                            # Resumen general
                            texto_total = " ".join(df['opinion'].astype(str).tolist())
                            resumen_general = resumir_texto(texto_total)
                            
                            # Temas principales
                            temas = extraer_temas(df['opinion'].astype(str).tolist(), top_n=7)
                            
                            # Mostrar resultados
                            st.markdown("#### 📌 Resumen General")
                            st.info(resumen_general)
                            
                            st.markdown("#### 🔍 Temas Detectados")
                            for i, tema in enumerate(temas, 1):
                                st.markdown(f"{i}. {tema.capitalize()}")
                            
                            # Análisis por sentimiento
                            st.markdown("#### 😃😐😠 Temas por Sentimiento")
                            for sent in ['Positivo', 'Neutro', 'Negativo']:
                                if sent in df['sentimiento'].values:
                                    textos = df[df['sentimiento'] == sent]['opinion']
                                    temas_sent = extraer_temas(textos.tolist(), top_n=3)
                                    st.markdown(f"**{sent}:** {', '.join(temas_sent)}")
                        except Exception as e:
                            st.error(f"Error en el análisis: {str(e)}")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")

# Sección para análisis de texto individual
st.subheader("✍️ Analizar Texto Individual")
user_input = st.text_area(
    "Escribe o pega un comentario para analizar:",
    height=150,
    placeholder="Ej: El producto es excelente, cumple con todas mis expectativas..."
)

if st.button("Analizar Comentario", key="analyze_single"):
    if user_input.strip():
        with st.spinner("Analizando..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment = classify_sentiment(user_input)
                st.metric("Sentimiento", sentiment)
            
            with col2:
                resumen = resumir_texto(user_input)
                st.text_area("Resumen", resumen, height=100)
            
            with col3:
                try:
                    keywords = kw_model.extract_keywords(
                        user_input, 
                        top_n=3, 
                        stop_words='spanish'
                    )
                    st.markdown("**Palabras clave:**")
                    for kw in keywords:
                        st.markdown(f"- {kw[0]} (score: {kw[1]:.2f})")
                except:
                    st.warning("No se pudieron extraer palabras clave")
    else:
        st.warning("Por favor ingresa un comentario para analizar")

# Footer
st.markdown("---")
st.caption("""
    Herramienta desarrollada para análisis de sentimientos - 
    [Documentación](https://www.example.com/docs) | 
    [Reportar problema](https://www.example.com/issues)
""")