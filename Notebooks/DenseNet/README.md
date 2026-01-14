# COVID-19 Radiography Classification with DenseNet Models

Ce projet présente l'utilisation de deux architectures de modèles **DenseNet** pour la classification des radiographies liées au COVID-19.
Nous avons testé les modèles suivants :

1. **DenseNet121**
2. **DenseNet169**

### Data Exploration

La **Data Exploration** a été réalisée pour les deux modèles et est enregistrée sous le fichier **"Exploration_Dataset_Covid.ipynb"**. 
Ce notebook contient les étapes de traitement, d'analyse et de visualisation des données.

---

### Structure du Projet

Le projet est divisé en deux sous-dossiers, chacun contenant un modèle pré-entraîné DenseNet (DenseNet121 ou DenseNet169) et son notebook Python associé.

#### Dossier **`DenseNet121/`** :

Dans ce dossier, vous trouverez les deux fichiers suivants :
- **cnn_densenet121_modele_masksh5.keras** : Le modèle pré-entraîné DenseNet121 sauvegardé.
- **DenseNet121.ipynb** : Le notebook Python contenant le code du modèle DenseNet121, qui inclut les étapes suivantes :
  1. **Préprocessing** des données
  2. **Modélisation** du réseau DenseNet121
  3. **Évaluation du modèle** et interprétation des résultats.

#### Dossier **`DenseNet169/`** :

Dans ce dossier, vous trouverez les deux fichiers suivants :
- **cnn_densenet169_modele.keras** : Le modèle pré-entraîné DenseNet169 sauvegardé.
- **DenseNet169.ipynb** : Le notebook Python contenant le code du modèle DenseNet169, qui inclut les étapes suivantes :
  1. **Préprocessing** des données
  2. **Modélisation** du réseau DenseNet169
  3. **Évaluation du modèle** et interprétation des résultats.

---

### Structure des Notebooks

Chaque notebook (**DenseNet121.ipynb** et **DenseNet169.ipynb**) suit une structure similaire et comprend les sections suivantes :

1. **Préprocessing** :
   - Chargement et nettoyage des données (images et labels).
   - Prétraitement des images (redimensionnement, normalisation, augmentation d'images).

2. **Modélisation** :
   - Construction du modèle DenseNet (DenseNet121 ou DenseNet169).
   - Compilation et entraînement du modèle sur les données.
   - Sauvegarde du modèle entraîné.

3. **Évaluation du modèle et interprétation des résultats** :
   - Calcul des métriques de performance du modèle (précision, rappel, F1-score, etc.).
   - Visualisation des courbes d'apprentissage (accuracy et loss).
   - Interprétation des résultats de classification.

---

### Installation des Dépendances

Pour exécuter ce projet, installez les dépendances en utilisant le fichier **`requirements.txt`**.
Vous pouvez installer toutes les bibliothèques nécessaires en exécutant la commande suivante :

```bash
pip install -r requirements.txt
