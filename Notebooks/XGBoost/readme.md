# Notebooks XGBoost – Projet COVID-19

Ce dossier regroupe l’ensemble des notebooks du **projet COVID-19**, organisés de 1 à 7.  
Ils couvrent toute la chaîne : analyse, création des datasets, preprocessing, tests de modèles, sélection et optimisation XGBoost, puis interprétation avec SHAP.

---

## 📘 1. Analyse du jeu de données
`1-Analyse.ipynb`

Après analyse du jeu de données, nous avons constaté que la **résolution des images** peut fortement influencer :

- la consommation mémoire,
- le temps de calcul,
- les performances du modèle.

Pour cela, l’étude propose de **créer plusieurs datasets** selon différentes tailles d’images afin d’observer l’impact de la résolution sur la précision et l’efficacité du modèle.

---

## 📘 2. Génération des datasets
`2-Génération_Dataset.ipynb`

Construction du dataset complet à partir des images brutes des **quatre classes**.  
Objectifs :

- créer plusieurs versions du dataset : **64×64, 128×128, etc.**
- permettre la comparaison des performances des modèles selon la résolution
- vérifier si une taille d’image donne un meilleur compromis entre vitesse et précision

---

## 📘 3. Pré-processing
`3-Pre-processing.ipynb`

Pour chaque taille d’image, plusieurs traitements ont été appliqués :

- **CLAHE** : amélioration du contraste pour faire ressortir les détails
- **HOG** : extraction des contours et de la structure globale de l’image
- **Split** du dataset en train/test
- **Normalisation** pour éviter toute fuite de données

Objectif : préparer les images pour une classification supervisée robuste.

---

## 📘 4. Test de plusieurs modèles
`4-Test_plusieurs_modeles.ipynb`

Évaluation initiale de plusieurs modèles de classification :

- Random Forest  
- Logistic Regression  
- XGBoost (
- Autres modèles supervisés

But : sélectionner le modèle le plus prometteur et comparer leur comportement sur les différentes résolutions d’images.

---

## 📘 5. Sélection du modèle
`5-Selection_modele.ipynb`

Après comparaison, **XGBoost** est retenu comme modèle principal.  
Cette étape comprend :

- Interpretation : Equilibré/Sur-apprentissage/Sous-apprentissage
- Filtrage pour Equilibrage faible/Solide
- Affinage (On garde les modèles avec un écart < 0.02)

Objectif : confirmer le choix du meilleur modèle avant optimisation.

---

## 📘 6. Optimisation XGBoost
`6-Optimisation_XGB`

Optimisation du modèle en deux approches :

### 🔍 1. Exploration  
- variation du **learning rate** pour comprendre la dynamique du modèle  
- observation des performances selon plusieurs valeurs

### ⚙️ 2. Exploitation (tuning)  
- variation du **max_depth**  
- application d’un **poids spécifique à la classe COVID** afin d’inciter le modèle à mieux prédire cette classe, en augmentant sa sensibilité sur les patients COVID positifs.
- objectif : **maximiser le F1-score**, particulièrement critique en diagnostic médical

---

## 📘 7. Analyse SHAP (Interprétabilité)
`7-Bonus_SHAP_XGB.ipynb`

Analyse avancée du modèle XGBoost avec SHAP :  
- **distribution des valeurs SHAP** (positives, négatives, neutres)  
- **SHAP global** (importance moyenne des features)  
- **summary_plot**  
- **force_plot** (impact des features sur une prédiction individuelle)

Objectif : comprendre comment chaque variable contribue à la prédiction COVID / non-COVID.

---

## 📦 Fichier supplémentaire
### `requirements.txt`
Liste des dépendances nécessaires pour exécuter l’ensemble des notebooks.

---

## 🎯 Objectif du dossier
Centraliser tous les notebooks du projet COVID, permettant de suivre clairement chaque étape :  
de l’analyse des données à l’interprétation finale du modèle optimisé XGBoost.

