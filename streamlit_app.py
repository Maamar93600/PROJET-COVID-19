# ==== LES IMPORTS ====
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
import os
import random
import shap

import xgboost as xgb
from skimage.feature import hog
from skimage import exposure

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ==== COULEURS DE L'APPLICATION ====
st.markdown("""
            <style>
            /*Couleur de fond*/
            html, body, [data-testid="stAppViewContainer"]{background-color: #E8F0F8 !important;}

            /*Sidebar*/
            section[data-testid="stSidebar"]{background-color: #D0E2F2 !important;}
            
            /*Titres*/
            h1,h2,h3{color: #0B1A33;
            font-weight: 700;}
            
            /*Texte*/
            p, li, span, label{color: #0B1A33 !important;}
            
            /*Boutons*/
            .stButton>button{background-color: #3A7CA5 !important;
            color: white !important;
            border-radius: 8px;
            border: none;}
            
            /*Encadrés*/
            div[data-testid="stMarkdown"]{color: #0B1A33;}
            </style>
            """, unsafe_allow_html=True)

# ==== Import des modèles ====
@st.cache_resource
def inceptionv3():
    return load_model("Modeles/inceptionV3.keras")
@st.cache_resource
def densenet121():
    return load_model("Modeles/cnn_densenet121_modele_masksh5.keras")
@st.cache_resource
def xgboost():
    return joblib.load("Modeles/xgb_model_best.joblib")
@st.cache_resource
def xgb_scaler():
    return joblib.load("Modeles/scaler.joblib")

# ==== DEFINITION DES FONCTIONS =====
mask_dir = ["Datas/COVID/masks", "Datas/Sain/masks", "Datas/Autres/masks"]

def preprocess_inceptionv3(uploaded_file):
    """
    Pipeline pour InceptionV3:
    -Lecture de l'image uploadée
    -Applique CLAHE
    -Resize de l'image en 256x256
    -Chercher et applique le mask correspondant
    -Normalise avec preprocess_input
    Retourne : 
    img_rgb_256 : image RGB en 256x256 masquée (pour affichage streamlit)
    filename : nom du ficher
    mask_path : chemin du mask
    """
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Impossible de décoder l'image uploadée.")
    filename = uploaded_file.name

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe,a,b))
    img_bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    img_bgr_256 = cv2.resize(img_bgr_clahe, (256,256))

    mask_path = None
    for d in mask_dir:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            mask_path = candidate
            break

    if mask_path is None:
        raise FileNotFoundError(f"Aucun mask nommé {filename} trouvé.")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise ValueError(f"Mask introuvable: {mask_path}")
    
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    mask_3c = cv2.merge([mask_bin, mask_bin, mask_bin])
    img_bgr_masked_256 = cv2.bitwise_and(img_bgr_256, mask_3c)

    img_rgb_256 = cv2.cvtColor(img_bgr_masked_256, cv2.COLOR_BGR2RGB)
    x = img_rgb_256.astype("float32")
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    return x, img_bgr_masked_256, filename, mask_path

def preprocess_xgb(uploaded_file):
    """
    Pipeline pour XGBoost:
    -Lecture image en GRAYSCALE
    -Resize en 128x128
    -Appliquer le mask
    -CLAHE (clipLimit=2, tileGridSize=(4,4))
    -HOG (orient=9, cell=12x12, block=2x2)
    -reshape (1, n_features)
    -StandardScaler
    Retourne : dtest : DMatrix pour XGBoost
    """
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.frombuffer(bytes_data, np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError("Impossible de décoder l'image uploadée.")
    filename = uploaded_file.name
    img_gray = cv2.resize(img_gray, (128,128))

    mask_path = None
    for d in mask_dir:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            mask_path = candidate
            break
    if mask_path is None:
        raise FileNotFoundError(f"Aucun mask nommé {filename} trouvé.")
    
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise ValueError(f"Mask introuvable: {mask_path}")
    
    if mask_gray.shape != img_gray.shape:
        mask_gray = cv2.resize(mask_gray, (128,128))
    
    _, mask_bin = cv2.threshold(mask_gray, 127, 155, cv2.THRESH_BINARY)
    img_masked = cv2.bitwise_and(img_gray, mask_bin)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (4,4))
    img_clahe = clahe.apply(img_masked)

    features = hog(img_clahe,
                   orientations=9,
                   pixels_per_cell=(12,12),
                   cells_per_block=(2,2),
                   block_norm='L2-Hys',
                   transform_sqrt=True,
                   feature_vector=True)
    
    x = features.reshape(1, -1)

    scaler = xgb_scaler()
    x_scaled = scaler.transform(x)
    dtest = xgb.DMatrix(x_scaled)

    return dtest, x_scaled

def preprocess_densenet121(uploaded_file):
    """
    Pipeline Pour DenseNet121:
    -Lecture image RGB
    -Resize 256x256
    -Appliquer mask correspondant
    -Normalisation en [0,1]
    -Resize final 224x224
    -expand_dim
    Retourne : x (1,224,224,3), img_masked, filename, mask_path
    """
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.frombuffer(bytes_data, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Impossible de décoder l'image uploadée.")
    
    filename = uploaded_file.name

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256,256))

    mask_path = None
    for d in mask_dir:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            mask_path = candidate
            break
    if mask_path is None:
        raise FileNotFoundError(f"Aucun mask nommé {filename} trouvé.")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask introuvable: {mask_path}")
    
    if mask.shape != (256,256):
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
    
    mask_bin = mask.astype(np.float32)/255.0

    mask_3c = np.stack([mask_bin, mask_bin, mask_bin], axis=-1)
    
    masked_img = img_rgb.astype(np.float32)*mask_3c
    masked_norm = np.clip(masked_img/255.0, 0.0, 1.0)

    masked_224 = cv2.resize(masked_norm, (224,224))

    x = np.expand_dims(masked_224,axis=0)
    return x, masked_norm, filename, mask_path

def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    """
    img : np.array shape (1,H,W,3) preprocessé
    pred_index : indice de la classe prédite
    """
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=0)

    last_conv_layer = model.get_layer(last_conv_layer_name)    
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img, training=False)
        if isinstance(preds, (list,tuple)):
            preds = preds[-1]
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)/(tf.reduce_max(heatmap)+1e-8)
    return heatmap.numpy()

def overlay_gradcam(heatmap, img, alpha=0.4):
    """
    heatmap : 2D(H,W)
    img : image RGB uint8 (H,W,3)
    """
    h,w = img.shape[:2]
    heatmap_resize = cv2.resize(heatmap, (w,h))
    heatmap_uint8 = np.uint8(255*heatmap_resize)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(alpha*heatmap_color + (1- alpha)*img)
    return superimposed

    
# ==== APPLICATION ====
st.sidebar.title("Sommaire")

pages = ["Accueil", "Covid-19", "Exploration des données",
          "XGBoost", "DenseNet121", "InceptionV3", "Tester les modèles", "Perspectives", "A propos"]

page = st.sidebar.radio("", pages)

# ==== PAGE ACCUEIL ====
if page == "Accueil":
    st.title("Détection du Covid-19 sur des radiographies pulmonaires")
    st.image("Images/accueil.jpeg", width=500)
    st.write("""
             Cette application a été réalisée dans le cadre d'un projet applicatif durant la formation DataScientest.
             L'objectif est d'enraîner un modèle capable de reconnaitre des patients atteints de Covid-19 à partir de radiographies.

             Trois modèles ont été élaborés et seront présentés dans cette application.

             Utiliser le menu de gauche pour naviguer entre :
             - Présentation rapide de la pathologie
             - L'exploration du jeu de données
             - La modélisation
             - Test des différents modèles
             """)
    
# ==== PAGE COVID-19 ====
if page == "Covid-19":
    st.title("Covid-19")
    st.write("""
             La pandémie de COVID-19, qui a débuté fin 2019, a exacerbé la pression sur les systèmes de
            santé mondiaux, révélant l'urgence de développer des méthodes de détection rapides et
            efficaces. Le diagnostic précoce est crucial pour limiter la propagation du virus, optimiser les
            traitements et éviter les surcharges hospitalières. Toutefois, malgré les avancées dans les tests
            diagnostiques, plusieurs défis demeurent dans la mise en œuvre de méthodes de détection
            efficaces, notamment dans les pays à faibles ressources et dans les environnements à forte
            demande.
             """)
    col1, col2 = st.columns(2)
    with col1:
        st.image("Images/covid_france.png",
                 caption="Nombre de décès causés par le Covid-19 en France",
                 use_container_width=True)
    with col2:
        st.image("Images/covid_monde.png",
                 caption="Nombre total de décès causés par le Covid-19 dans le monde",
                 use_container_width=True)
    st.write("Au printemps 2025, on dénombre 168 000 décès en France et 7 089 989 à l'échelle mondiale.")
        
    st.subheader("Avantages d'un modèle")
    st.write("""
            - Réduit les dépenses supplémentaires (Examens multiples)
            - Une seule personne suffit pour utiliser l'outil
            - Permet une détection plus rapide des cas lors d'une pandémie
             """)

# ==== PAGE EXPLORATION DES DONNEES ====
if page == "Exploration des données":
    st.title("Exploration des données")
    st.write("""
             Le jeu de données est composé d'images radiographiques et des masques correspondants.
             Les données sont issues de plusieurs sources publiques et regrouper sur Kaggle.<sup>[1]</sup>
             La structure de l'organisation de ces dernière est la suivante :
             """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("Images/structure_data.png",
                caption="Structure du jeu de données",
                width=250)
    # ---- Histogramme ----    
    categories = ["Covid-19", "Opacité pulmonaire", "Normal", "Pneumonie virale"]
    nb_img = [3616, 6012, 10192, 1345]
    nb_mask = [3616, 6012, 10192, 1345]

    hist = pd.DataFrame({"Catégories" : categories,
                        "Images" : nb_img,
                        "Masques" : nb_mask})
    hist_long = hist.melt(id_vars = "Catégories",
                          value_vars = ["Images", "Masques"],
                          var_name = "Type",
                          value_name= "Nombre")
    fig = px.bar(hist_long, x="Catégories",
                 y="Nombre",
                 color="Type",
                 barmode="group",
                 title="Répartition des données par catégorie")
    
    fig.update_layout(xaxis_title="Catégories",
                      yaxis_title="Nombre",
                      legend_title="",
                      template="simple_white",
                      plot_bgcolor="#f0f4f8",
                      paper_bgcolor="#f0f4f8")
    
    fig.update_traces(marker_line_width=5,
                      text=hist_long["Nombre"],
                      textposition="inside")
    fig.for_each_trace(lambda t: t.update(marker_color="#113A69") if t.name =="Images" else t.update(marker_color="#ce519a"))

    st.subheader("Répartition des données par catégorie")
    st.write("""
            - Disparté dans le nombre d'images par catégorie
            - Nombre d'images et nombre de masques égaux
            - Pas de fichiers manquants ou corrompu
             """)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Taille et formats des fichiers")
    rows = [categories[:2], categories[2:]]
    for row in rows:
        cols = st.columns(2)
        for col, cat in zip(cols, row):
            with col:
                st.markdown(f"""
                            **{cat}**
                            - Images : PNG - 299x299 px
                            - Masques : PNG - 256x256 px
                            """)
    # ---- Figure taille/formats ----
    fig = go.Figure()
    # ----Grand carré----
    fig.add_shape(type="rect",
                  x0=0,
                  y0=0,
                  x1=1,
                  y1=1,
                  line=dict(width=0),
                  fillcolor="rgba(0,63,140,0.85)")
    #----Ombre----
    fig.add_shape(type="rect",
                  x0=0.04, y0=0.04,
                  x1=0.84, y1=0.84,
                  line=dict(width=0),
                  fillcolor="rgba(0,0,0,0.35)")
    #----Petit carré----
    fig.add_shape(type="rect",
                  x0=0.02, y0=0.02,
                  x1=0.82, y1=0.82,
                  line=dict(width=0),
                  fillcolor="rgba(255,122,200,1)")
    # ---- Légende ----
    fig.add_annotation(x=0.5, y=0.92,
                       text="Images<br>299x299 px - PNG",
                       showarrow=False,
                       font=dict(size=16, color="#90b0d6"))
    fig.add_annotation(x=0.4,
                       y=0.4,
                       text="Masques<br>256x256 px - PNG",
                       showarrow=False,
                       font=dict(size=14, color="#7a0040"))

    fig.update_xaxes(visible=False, range=[0,1])
    fig.update_yaxes(visible=False, range=[0,1], scaleanchor="x", scaleratio=1)

    fig.update_layout(height=300,
                      width=300,
                      margin=dict(l=10,r=10,t=10,b=10),
                      plot_bgcolor="#E8F0F8",
                      paper_bgcolor="#E8F0F8")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Echantillon d'images ----
    def charger_img(base_dir):
        """
        base_dir doit contenir 3 sous-dossiers
        """
        classes = ["COVID", "Sain", "Autres"]
        data ={}
        for cls in classes:
            dossier = os.path.join(base_dir, cls, "images")
            if os.path.join(dossier):
                fichiers = [os.path.join(dossier,f) for f in os.listdir(dossier)]
                data[cls] = fichiers
        return data

    def carrousel_aleatoire(images_dict, n=4):
        """
        Selectionne n images alétoires avec leur étiquettes
        """
        tirage = []
        for n in range(n):
            cls = random.choice(list(images_dict.keys()))
            img_path = random.choice(images_dict[cls])
            tirage.append((img_path,cls))
        return tirage
    
    if "echantillon" not in st.session_state:
        st.session_state.echantillon = []
    if "idx_img" not in st.session_state:
        st.session_state.idx_img = []

    st.subheader("Visualisation d'un échantillon d'images du dataset")
    images_dict = charger_img("Datas")
    if st.button("Générer un échantillon aléatoire"):
        st.session_state.echantillon = carrousel_aleatoire(images_dict, n=4)
        st.session_state.idx_img = 0

    if st.session_state.echantillon:
        col_prev, col_img, col_next = st.columns([1,3,1])

        with col_prev:
            if st.button("⬅️Précédent"):
                st.session_state.idx_img = (st.session_state.idx_img-1)% len(st.session_state.echantillon)
        with col_next:
            if st.button("➡️Suivant"):
                st.session_state.idx_img = (st.session_state.idx_img+1)% len(st.session_state.echantillon)
        with col_img:
            img_path, cls= st.session_state.echantillon[st.session_state.idx_img]
            img = Image.open(img_path)
            st.image(img, caption=f"Classe {cls}", width=350, use_container_width=True)
            st.markdown(f"<p style='text-align:center;margin-top:O.5rem;'>Image {st.session_state.idx_img+1}/{len(st.session_state.echantillon)}</p>",
                        unsafe_allow_html=True)           
    else:
        st.info("Cliquez sur le bouton pour afficher un échantillon")
    

# ==== PAGE XGBOOST ====
if page == "XGBoost":
    st.title("XGBoost")
    st.write("""
            Le choix du modèle XGBoost<sup>[2]</sup> a été défini après une série de tests réalisés sur 
            plusieurs classifieurs (_cf._ le rapport).
            """, unsafe_allow_html=True)

    onglet1, onglet2, onglet3 = st.tabs(["Pré-traitement", "Entraînement", "Evaluation"])
    with onglet1:
        st.header("Pré-traitement")
        st.subheader("Répartition des données")
        cola, colb = st.columns(2)
        with cola:
            st.write("""
                    Les données sont réparties en trois classes :
                    - 0 → Normal : 3610 images
                    - 1 → Covid-19 : 3610 images
                    - 2 → Autres (Opacité pulmonaire + Pneumonie virale) : 3610 images
                    """)
            st.write("""
                    Après traitement, les images ont été séparées en 2 ensembles :
                    - Ensemble d'entraînement : 80%
                    - Ensemble de test : 20%
                    """)
        with colb:
            cat = ["Normal", "Covid-19", "Autres"]
            total_par_classe = [3610, 3610, 3610]

            rows = []
            for c, total in zip(cat, total_par_classe):
                train = int(total*0.8)
                test = total-train
                rows.append({"Catégorie" : c, "Ensemble" : "Train", "Valeur": train})
                rows.append({"Catégorie" : c, "Ensemble" : "Test", "Valeur": test})
                
                df = pd.DataFrame(rows)

                fig = px.sunburst(df, path=["Catégorie","Ensemble"],
                                  values="Valeur")
            fig.update_layout(paper_bgcolor="#E8F0F8",
                              plot_bgcolor="#E8F0F8",
                              margin=dict(l=0,r=0,t=30,b=0),
                              height=300,
                              width=300)
            st.plotly_chart(fig, width=300)
                           
        st.write("Les données ont été enregistrées au format _.pkl_.")

        st.subheader("Exploration visuelle : CLAHE & HOG")
        img_gray = cv2.imread("Datas/COVID/images/COVID-32.png", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread("Datas/COVID/masks/COVID-32.png", cv2.IMREAD_GRAYSCALE)
        mask_resize = cv2.resize(mask,(299,299))

        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            st.image(img_gray, caption="Image originale",
                    channels="GRAY",
                    width=350)
        colx, coly, colz = st.columns(3)
        with colx:
            clip_limit = st.slider("Clip Limit", 1.0, 5.0, 2.0, 1.0)
            tile_size = st.slider("tileGridSize", 4, 16, 8, 2)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size,tile_size))
            img_clahe = clahe.apply(img_gray)
        with coly:
            show_mask = st.checkbox("Ajouter le masque", value=False)
            
        with colz:
            show_hog = st.checkbox("Afficher la représentation HOG", value=False)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_clahe, caption=f"Après CLAHE (clipLimit={clip_limit}, tileGridSize=({tile_size}x{tile_size})",
                     channels="GRAY", use_container_width=True)
        with col2:
            if show_mask:
                img_work = cv2.bitwise_and(mask_resize, img_clahe)
                st.image(img_work, caption="Image avec masque appliqué", use_container_width=True)
            else:
                img_work = img_clahe
                st.info("Cochez la case pour appliquer le masque")
        with col3:
            if show_hog:
                _, hog_img = hog(img_work,
                                 orientations=9,
                                 pixels_per_cell=(12,12),
                                 cells_per_block=(2,2),
                                 visualize=True,
                                 channel_axis=None)
                hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0,hog_img.max()))
                st.image(hog_img_rescaled, caption="HOG sur l'image CLAHE", use_container_width=True)
            else:
                st.info("Cochez la case pour afficher le HOG")

        
    with onglet2:
        st.header("Entraînement")
        st.write("Cette approche a permis de tester différentes vitesses d’apprentissage, " \
        "mais a été limitée par le temps et la complexité du planning de LR.")
        st.image("Images/loss_ml.png",
                caption="Courbe de logloss du modèle XGBoost",
                use_container_width=True)
        st.write("""
                Optimisation du modèle sur les paramètres suivants :
                - La valeur du _max_depth_
                - Amplification des poids de la classe Covid-19
                - La recherche du seuil F1 optimal pour la classe Covid-19
                """)

        st.write("""
                Après optimisation, les paramètres retenus sont :
                - _max_depth_ = 6
                - Facteur d'amplification du poids de la classe Covid-19 = 1.5
                - Seuil = 0.412406
                """)
        st.write("L'entraînement final sur 300 arbres conduit aux résultats suivants :")
    
        categories = ["Normal","Covid-19","Autres"]
        score_avant = {"Normal" : {"Précision":80.7,"Rappel":83.2,"F1-Score":81.9},
                        "Covid-19" : {"Précision":74.9,"Rappel":79.8,"F1-Score":77.2},
                        "Autres" : {"Précision":84.9,"Rappel":76.6,"F1-Score":80.5}}
            
        score_apres = {"Normal" : {"Précision":81.1,"Rappel":81.8,"F1-Score":81.4},
                        "Covid-19" : {"Précision":73.1,"Rappel":83.7,"F1-Score":78.1},
                        "Autres" : {"Précision":87.7,"Rappel":74.2,"F1-Score":80.4}}
            
        rows=[]
        for cat in categories:
            for metrique in score_avant[cat].keys():
                    rows.append({"Catégories":cat, "Métrique":metrique, "Avant":score_avant[cat][metrique],
                                 "Après":score_apres[cat][metrique]})
        df = pd.DataFrame(rows).melt(id_vars=["Catégories","Métrique"],
                                     value_vars=["Avant","Après"],
                                     var_name="Version",
                                     value_name="Valeur")

        fig = px.bar(df, x="Métrique", y="Valeur", color="Version",
                    barmode="stack",
                    facet_col="Catégories",
                    category_orders={"Métrique":["Précision","Rappel","F1-Score"]},
                    title="Scores par classe avant et après optimisation")
            
        fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
        fig.update_layout(yaxis=dict(range=[0,200], title="Score"),
                        bargap=0.05,
                        plot_bgcolor="#E8F0F8",
                        paper_bgcolor="#E8F0F8",
                        margin=dict(t=80,r=30,l=40,b=40))
        
        st.plotly_chart(fig, use_container_width=True)         
    

        col1,col2 = st.columns(2)
        with col1:
            cm = np.array([[0.83,0.12,0.04],
                       [0.11,0.80,0.09],
                       [0.09,0.14,0.77]])
            labels = ["Normal","Covid-19","Autres"]
            fig_cm = px.imshow(cm*100, x=labels, y=labels,
                           text_auto=".1f",
                           color_continuous_scale="PuBu",
                           labels=dict(x="Classe prédite", y="Classe réelle", color="Score"))
            fig_cm.update_layout(title="Avant",
                             xaxis_side="top",
                             margin=dict(l=30,r=30,t=60,b=30),
                             paper_bgcolor="#E8F0F8",
                             plot_bgcolor="#E8F0F8")
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            cm = np.array([[0.82,0.14,0.04],
                       [0.10,0.84,0.06],
                       [0.09,0.17,0.74]])
            labels = ["Normal","Covid-19","Autres"]
            fig_cm = px.imshow(cm*100, x=labels, y=labels,
                           text_auto=".1f",
                           color_continuous_scale="PuBu",
                           labels=dict(x="Classe prédite", y="Classe réelle", color="Score"))
            fig_cm.update_layout(title="Après",
                             xaxis_side="top",
                             margin=dict(l=30,r=30,t=60,b=30),
                             paper_bgcolor="#E8F0F8",
                             plot_bgcolor="#E8F0F8")
            st.plotly_chart(fig_cm, use_container_width=True)


    with onglet3:
        st.header("Evaluation")
        st.write("L'évaluation du modèle sur l'ensemble de test produit les performances suivantes :")

        data={"Catégories" : ["Normal","Covid-19","Autres"],
              "Précision" : [80.2,69.3,85.9],
              "Rappel" : [79.8,81.3,71.2],
              "F1-Score" : [80.0,74.8,78.0]}
        accuracy = 77.5

        df = pd.DataFrame(data).melt(id_vars="Catégories",
                                     value_vars=["Précision","Rappel","F1-Score"],
                                     var_name="Métriques",
                                     value_name="Valeur")
        fig = px.bar(df, x="Métriques",
                     y="Valeur",
                     color="Catégories",
                     barmode="group",
                     text="Valeur",
                     title="Rapport de classification durant le test")
        fig.update_traces(texttemplate="%{text:.f}", textposition="outside")
        fig.add_annotation(text=f"<b>Accuracy globale :</b> {accuracy:.1f}%",
                           xref="paper", yref="paper",
                           x=1.3, y=1.1,
                           showarrow=False,
                           font=dict(size=14, color="black"))
        fig.update_layout(yaxis=dict(range=[0,100], title="Score"),
                          xaxis_title="Métriques",
                          legend_title="Catégories",
                          legend=dict(y=0.75),
                          margin=dict(t=80,r=150),
                          bargroupgap=0.05,
                          plot_bgcolor="#E8F0F8",
                          paper_bgcolor="#E8F0F8")
        st.plotly_chart(fig, use_container_width=True)

        cm = np.array([[0.797784,0.152355,0.049861],
                       [0.119114,0.813019,0.067867],
                       [0.077562,0.207756,0.7146]])
        labels = ["Normal","Covid-19","Autres"]
        fig_cm = px.imshow(cm*100, x=labels, y=labels,
                           text_auto=".1f",
                           color_continuous_scale="PuBu",
                           labels=dict(x="Classe prédite", y="Classe réelle", color="Score"))
        fig_cm.update_layout(title="Matrice de confusion",
                             xaxis_side="top",
                             height=400,
                             width=400,
                             margin=dict(l=30,r=30,t=60,b=30),
                             paper_bgcolor="#E8F0F8",
                             plot_bgcolor="#E8F0F8")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.write("L'outils SHAP permet d'analyser plus précisement les prises de décision du modèle :")
        st.image("Images/shap_xgb.png",
                caption="SHAP du modèle XGBoost",
                use_container_width=True)



# ==== PAGE DENSENET121 ====
if page == "DenseNet121":
    st.title("DenseNet121")
    st.write("""
            Le modèle DenseNet121<sup>[3]</sup> est un réseau de neurones convolutifs profond basé sur des connexions densifiées entre les couches.
            Cela permet une meilleure propagation de l'information, une réduction du nombre de paramètres et des performances
            élevées en classification d'images.
             """, unsafe_allow_html=True)

    onglet1, onglet2, onglet3 = st.tabs(["Pré-traitement", "Entraînement", "Evaluation"])
    with onglet1:
        st.header("Pré-traitement")
        st.write("""
                Les données sont réparties en quatre classes :
                - 0 → Covid-19 : 3616 images
                - 1 → Opacité pulmonaire : 6012 images
                - 2 → Normal : 10192 images
                - 3 → Pneumonie virale : 1345 images
                """)
        st.write("""
                Après traitement, les images ont été séparées en 3 ensembles :
                - Ensemble d'entraînement : 68%
                - Ensemble de validation : 12%
                - Ensemble de test : 20%
                """)
        cat = ["Covid-19", "Opacité pulmonaire", "Normal", "Pneumonie virale"]
        total_par_classe = [3616, 6012, 10192, 1345]

        rows = []
        for c, total in zip(cat, total_par_classe):
            train = int(total*0.8)
            val = int(train*0.15)
            test = total-train
            rows.append({"Catégorie" : c, "Ensemble" : "Train", "Valeur": train})
            rows.append({"Catégorie" : c, "Ensemble" : "Validation", "Valeur": val})
            rows.append({"Catégorie" : c, "Ensemble" : "Test", "Valeur": test})
            
            df = pd.DataFrame(rows)

            fig = px.sunburst(df, path=["Catégorie","Ensemble"],
                                  values="Valeur")
        fig.update_layout(paper_bgcolor="#E8F0F8",
                              plot_bgcolor="#E8F0F8",
                              margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.write("L'equilibre des classes a été traité durant l'élaboration du modèle.")
        
    with onglet2:
        st.header("Entraînement")
        st.write("""
                L'entraînement du modèle a été réalisé en dégelant les 50 dernières couches du modèle.
                Les paramètres retenus pour la compilation de ce dernier sont :
                - Optimiseur : Adam (_learning_rate_ = 1.10<sup>-5</sup>)
                - Fonction de perte : "_Sparse categorical crossentropy_"
                - Metrique : accuracy
                - Callbacks : _EarlyStopping_, _ModelCheckpoint_ et _ReduceLROnPlateau_.
                - Pondération des classes : _compute_class_weight_.
                 """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image("Images/dense_loss.png",
                     caption="Evolution de la loss durant l'entraînement",
                     use_container_width=True)
        with col2:
            st.image("Images/dense_accuracy.png",
                     caption="Evolution de l'accuracy durant l'enraînement",
                     use_container_width=True)
    with onglet3:
        st.header("Evaluation")
        st.write("Après évaluation du modèle sur l'ensemble de test, les performances sont :")
        data={"Catégories" : ["Covid-19","Opacité pulmonaire","Normal","Pneumonie virale"],
              "Précision" : [76,85,89,89],
              "Rappel" : [76,82,90,92],
              "F1-Score" : [76,84,90,92]}
        accuracy = 86

        df = pd.DataFrame(data).melt(id_vars="Catégories",
                                     value_vars=["Précision","Rappel","F1-Score"],
                                     var_name="Métriques",
                                     value_name="Valeur")
        fig = px.bar(df, x="Métriques",
                     y="Valeur",
                     color="Catégories",
                     barmode="group",
                     text="Valeur",
                     title="Rapport de classification durant le test")
        fig.update_traces(texttemplate="%{text:.f}", textposition="outside")
        fig.add_annotation(text=f"<b>Accuracy globale :</b> {accuracy:.1f}%",
                           xref="paper", yref="paper",
                           x=1.3, y=1.1,
                           showarrow=False,
                           font=dict(size=14, color="black"))
        fig.update_layout(yaxis=dict(range=[0,100], title="Score"),
                          xaxis_title="Métriques",
                          legend_title="Catégories",
                          legend=dict(y=0.75),
                          margin=dict(t=80,r=150),
                          bargroupgap=0.05,
                          plot_bgcolor="#E8F0F8",
                          paper_bgcolor="#E8F0F8")
        st.plotly_chart(fig, use_container_width=True)

        cm = np.array([[0.76,0.11,0.12,0.01],
                       [0.08,0.82,0.10,0.00],
                       [0.04,0.04,0.90,0.01],
                       [0.00,0.01,0.03,0.96]])
        labels = ["Covid-19","Opacité pulmonaire","Normal","Pneumonie viral"]
        fig_cm = px.imshow(cm*100, x=labels, y=labels,
                           text_auto=".1f",
                           color_continuous_scale="PuBu",
                           labels=dict(x="Classe prédite", y="Classe réelle", color="Score"))
        fig_cm.update_layout(title="Matrice de confusion",
                             xaxis_side="top",
                             height=400,
                             width=400,
                             margin=dict(l=30,r=30,t=60,b=30),
                             paper_bgcolor="#E8F0F8",
                             plot_bgcolor="#E8F0F8")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.image("Images/roc_densenet121.png",
                caption="Courbe ROC/AUC du modèle DenseNet121",
                use_container_width=True)
        
        st.subheader("Echantillon d'images Covid-19 mal classées")
        st.image("Images/erreurs_densenet121.png", use_container_width=True)


# ==== PAGE INCEPTIONV3 ====
if page == "InceptionV3":
    st.title("InceptionV3")
    st.write("""
            Cette partie est inspirée du travail réalisé par Alqatani _et al._<sup>[4]</sup>.
            Ils utilisent le modèle InceptionV4 pour la détection du Covid-19 à partir de radiographies thoraciques,
            qui a conduit à d'excellents résultats.
            Nous avons, donc, adapté leur méthodologie à notre jeu de données sur le modèle InceptionV3<sup>[5]</sup>.
            """, unsafe_allow_html=True)


    onglet1, onglet2, onglet3 = st.tabs(["Pré-traitement", "Entraînement", "Evaluation"])
    with onglet1:
        st.header("Pré-traitement")
        st.write("""
                Les données sont réparties en trois classes :
                - 0 → Sain : 3500 images
                - 1 → Covid-19 : 3500 images
                - 2 → Autres (Opacité pulmonaire + Pneumonie virale) : 3500 images
                """)
        st.write("""
                Après traitement, les images ont été séparées en 3 ensembles :
                - Ensemble d'entraînement : 70%
                - Ensemble de validation : 15%
                - Ensemble de test : 15%
                """)
        cat = ["Normal", "Covid-19", "Autres"]
        total_par_classe = [3610, 3610, 3610]

        rows = []
        for c, total in zip(cat, total_par_classe):
            train = int(total*0.85)
            val = int(train*0.176)
            test = total-train
            rows.append({"Catégorie" : c, "Ensemble" : "Train", "Valeur": train})
            rows.append({"Catégorie" : c, "Ensemble" : "Validation", "Valeur": val})
            rows.append({"Catégorie" : c, "Ensemble" : "Test", "Valeur": test})
                
            df = pd.DataFrame(rows)

            fig = px.sunburst(df, path=["Catégorie","Ensemble"],
                                  values="Valeur")
        fig.update_layout(paper_bgcolor="#E8F0F8",
                              plot_bgcolor="#E8F0F8",
                              margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Exploration visuelle : CLAHE & Masque")
        img_gray = cv2.imread("Datas/COVID/images/COVID-32.png", cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread("Datas/COVID/masks/COVID-32.png", cv2.IMREAD_GRAYSCALE)
        mask_resize = cv2.resize(mask, (299,299))

        col1,col2 = st.columns([2,1])
        with col1:
            clip_limit = st.slider("Clip Limit", 1.0, 5.0, 2.0, 1.0)
            tile_size = st.slider("tileGridSize", 4, 16, 8, 2)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size,tile_size))
            img_clahe = clahe.apply(img_gray)
        with col2:
            show_mask = st.checkbox("Ajouter le masque")

        col1,col2, col3 = st.columns(3)
        with col1:
            st.image(img_gray, caption="Image originale",
                 channels="GRAY", use_container_width=True)
        with col2:
            st.image(img_clahe, caption=f"Après CLAHE (clipLimit={clip_limit}, tileGridSize=({tile_size}x{tile_size})",
                     channels="GRAY", use_container_width=True)
        with col3:
            if show_mask:
                img_mask = cv2.bitwise_and(mask_resize, img_clahe)
                st.image(img_mask, caption="Image avec masque appliqué", use_container_width=True)
            else:
                st.info("Cochez la case pour appliquer le masque")

    with onglet2:
        st.header("Entraînement")
        st.write("""
                L'entraînement du modèle a été réalisé en dégelant les 20 dernières couches du modèle.
                Les paramètres retenus pour la compilation de ce dernier sont :
                - Optimiseur : SGD (_learning_rate_ = 1.10<sup>-4</sup> et momentum=0.9)
                - Fonction de perte : "_Sparse categorical crossentropy_"
                - Metrique : accuracy
                - Callbacks : _EarlyStopping_, _ModelCheckpoint_ et _ReduceLROnPlateau_.
                 """, unsafe_allow_html=True)
        
        st.image("Images/inception_accuracy_loss_20layers.png",
                     caption="Evolution de la loss et de l'accuracy durant l'entraînement (20 premères epochs)",
                     use_container_width=True)
        
        st.image("Images/inception_accuracy_loss_20layers2.png",
                     caption="Evolution de l'accuracy et de la loss durant l'enraînement (ajout 80 epochs)",
                     use_container_width=True)

    with onglet3:
        st.header("Evaluation")
        st.write("Après évaluation du modèle sur l'ensemble de test, les performances sont :")
        data={"Catégories" : ["Sain","Covid-19","Autres"],
              "Précision" : [82,81,86],
              "Rappel" : [88,80,81],
              "F1-Score" : [85,81,83]}
        accuracy = 83

        df = pd.DataFrame(data).melt(id_vars="Catégories",
                                     value_vars=["Précision","Rappel","F1-Score"],
                                     var_name="Métriques",
                                     value_name="Valeur")
        fig = px.bar(df, x="Métriques",
                     y="Valeur",
                     color="Catégories",
                     barmode="group",
                     text="Valeur",
                     title="Rapport de classification durant le test")
        fig.update_traces(texttemplate="%{text:.f}", textposition="outside")
        fig.add_annotation(text=f"<b>Accuracy globale :</b> {accuracy:.1f}%",
                           xref="paper", yref="paper",
                           x=1.3, y=1.1,
                           showarrow=False,
                           font=dict(size=14, color="black"))
        fig.update_layout(yaxis=dict(range=[0,100], title="Score"),
                          xaxis_title="Métriques",
                          legend_title="Catégories",
                          legend=dict(y=0.75),
                          margin=dict(t=80,r=150),
                          bargroupgap=0.05,
                          plot_bgcolor="#E8F0F8",
                          paper_bgcolor="#E8F0F8")
        st.plotly_chart(fig, use_container_width=True)

        cm = np.array([[0.8762,0.080,0.0438],
                       [0.1108,0.8019,0.0876],
                       [0.0838,0.1105,0.8057]])
        labels = ["Sain","Covid-19","Autres"]
        fig_cm = px.imshow(cm*100, x=labels, y=labels,
                           text_auto=".1f",
                           color_continuous_scale="PuBu",
                           labels=dict(x="Classe prédite", y="Classe réelle", color="Score"))
        fig_cm.update_layout(title="Matrice de confusion",
                             xaxis_side="top",
                             height=400,
                             width=400,
                             margin=dict(l=30,r=30,t=60,b=30),
                             paper_bgcolor="#E8F0F8",
                             plot_bgcolor="#E8F0F8")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.image("Images/roc_auc_inceptionv3.png",
                caption="Courbe ROC/AUC du modèle DenseNet121",
                use_container_width=True)
        st.subheader("Echantillon d'images Covid-19 mal classées")
        st.image("Images/erreurs_inceptionv3.png", use_container_width=True)

# ==== PAGE TESTER LE MODELE ====
if page == "Tester les modèles":
    st.title("Tester les modèles")
    uploaded_file = st.file_uploader("Choisir une radiographie pulmonaire", type=["jpg", "jpeg", "png"])

    model_choice = st.selectbox("Choisir un modèle à tester:", 
                            ["XGBoost", "DenseNet121", "InceptionV3"])
    if model_choice == "XGBoost":
        model = xgboost()
    elif model_choice == "DenseNet121":
        model = densenet121()
    else:
        model = inceptionv3()
   
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file, caption="Image chargée", use_container_width=300)

    if st.button("Prédire"):
        if uploaded_file is not None:
            progress = st.progress(0)
            if model_choice == "InceptionV3":
                progress.progress(10)
                x, img_masked, filename, mask_path = preprocess_inceptionv3(uploaded_file)
                progress.progress(30)
            elif model_choice == "XGBoost":
                progress.progress(10)
                x, x_scaled = preprocess_xgb(uploaded_file)
                progress.progress(30)
            elif model_choice == "DenseNet121":
                progress.progress(10)
                x, masked_norm, filename, mask_path = preprocess_densenet121(uploaded_file)
                progress.progress(30)          

        
            progress.progress(75)
            preds = model.predict(x)
            progress.progress(100)
            if model_choice == "XGBoost":
                prob_xgb = preds[0]
                pred_class = int(np.where(prob_xgb[1] >= 0.412406, 1, np.argmax(prob_xgb)))
            else:                
                pred_class = np.argmax(preds)
            
            prob_predite = preds[0][pred_class]    
            prob = round(float(prob_predite)*100, 2)

            if model_choice in ["XGBoost", "InceptionV3"]:

                classes = ["Sain", "Covid-19", "Autres"]

                st.subheader("Résultat de la prédiction")
                st.write(f"**Classe prédite :** {classes[pred_class]}")
                st.write("**Probabilité :**", prob, "%")

                probs = (preds[0]*100).round(2)
                df_probs = pd.DataFrame({"Classe" : classes,
                                    "Probabilité (%)" : probs})
            
                fig_pie = px.pie(df_probs,
                            names = "Classe",
                            values = "Probabilité (%)",
                            hole = 0.4)
            
                fig_pie.update_traces(textposition = "outside",
                            textinfo = "percent+label",
                            textfont_size = 14)
            
                fig_pie.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                            plot_bgcolor = "rgba(0,0,0,0)",
                            font = dict(color="#0B1A33"),
                            legend = dict(font=dict(size=18)),
                            height= 400)
            
                st.plotly_chart(fig_pie, use_container_width=True)
                st.dataframe(df_probs, use_container_width=True)

                if model_choice == "XGBoost":
                    st.subheader("SHAP (force plot)")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(x_scaled)

                    class_names=["Sain","Covid-19","Autres"]
                    n_classes = shap_values.shape[2]

                    for i, class_name in enumerate(class_names):
                        st.markdown(f"### Classe : **{class_name}**")

                        shap_vals_for_class = shap_values[0,:,i]
                        expected_value = explainer.expected_value[i]

                        shap.force_plot(expected_value,
                                    shap_vals_for_class,
                                    x_scaled[0,:],
                                    matplotlib=True,
                                    show=False)
                        fig = plt.gcf()
                        ax = plt.gca()

                        fig.patch.set_facecolor("#E8F0F8")
                        ax.set_facecolor("#E8F0F8")

                        st.pyplot(fig)
                        plt.clf()

            else:
                classes = ["Covid-19", "Opacité pulmonaire", "Normal", "Pneumonie Virale"]

                st.subheader("Résultat de la prédiction")
                st.write(f"**Classe prédite :** {classes[pred_class]}")
                st.write("**Probabilité :**", prob, "%")

                probs = (preds[0]*100).round(2)
                df_probs = pd.DataFrame({"Classe" : classes,
                                    "Probabilité (%)" : probs})
            
                fig_pie = px.pie(df_probs,
                            names = "Classe",
                            values = "Probabilité (%)",
                            hole = 0.4)
            
                fig_pie.update_traces(textposition = "outside",
                            textinfo = "percent+label",
                            textfont_size = 12)
            
                fig_pie.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                            plot_bgcolor = "rgba(0,0,0,0)",
                            font = dict(color="#0B1A33"),
                            legend = dict(font=dict(size=18)),
                            height= 400)
            
                st.plotly_chart(fig_pie, use_container_width=True)
                st.dataframe(df_probs, use_container_width=True)

                st.subheader("Grad-CAM")

                last_conv_name = "conv5_block16_concat"
                heatmap = make_gradcam_heatmap(x, model, last_conv_name, pred_index=pred_class)
                img_base = (masked_norm*255).astype("uint8")

                overlay = overlay_gradcam(heatmap, img_base)

                st.image(overlay, caption="Carte Grad-CAM du modèle DenseNet121", width=400)
                
        else:
            st.error("Veuillez importer une image avant de procéder à la prédiction.")
            st.stop()

# ==== PAGE PERSPECTIVES ====
if page == "Perspectives":
    st.title("Perspectives")
    st.subheader("Améliorations")
    st.write("""
            XGBoost :
            - Optimisation de paramètres
            - Augmentation des données
            - Reste limité pour une classification sur des images
             """)
    st.write("""
             DenseNet121 & InceptionV3 :
            - Optimisation de paramètres
            - Changer d'optimiseur
            - Dégèle de plus ou moins de couches
            - Augmentation des données
             """)
    st.subheader("Autres possibilités")
    st.write("""
            - Entraîner d'autres modèles CNN (EfficientNetB7,...)
            - Combinaisons de modèles : CNN + CNN ou XGBoost + CNN
             """)

# ==== PAGE A PROPOS ====
if page == "A propos":
    st.title("A propos")

    st.subheader("Les auteurs")
    st.write("Asma KERKACHE")
    st.write("Maamar BENHENNI")
    st.write("Richard TONNANG")

    st.subheader("Remerciements")
    st.write("""
            - Toutes les équipes de Datascientest
            - Chef de cohorte : Vincent
            - Mentor de projet : Antoine
             """)

    st.subheader("Bibliographie")
    st.write("[1] T. Rahman, M. Chowdhury, et A. Khandakar, « Covid-19 radiography database ». 2021. "
            "[En ligne]. Disponible sur: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data")
    st.write("[2] T. Chen et C. Guestrin, « XGBoost: A Scalable Tree Boosting System »,"
            " in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, " \
            "San Francisco California USA: ACM, août 2016, p. 785-794. doi: 10.1145/2939672.2939785.")
    st.write( "[3] G. Huang, Z. Liu, L. van der Maaten, et K. Q. Weinberger, « Densely Connected Convolutional Networks », 2016," \
            " arXiv. doi: 10.48550/ARXIV.1608.06993.")
    st.write("[4] A. Alqahtani et al., « A Transfer Learning Based Approach for COVID-19 Detection Using Inception-v4 Model »," \
    "Intelligent Automation & Soft Computing, vol. 35, no 2, p. 1721-1736, 2023, doi: 10.32604/iasc.2023.025597.")
    st.write("[5] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, et Z. Wojna, « Rethinking the Inception Architecture"
    " for Computer Vision », 2015, arXiv. doi: 10.48550/ARXIV.1512.00567.")
