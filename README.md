![Profile](https://via.placeholder.com/150x150/4A90E2/FFFFFF?text=VOTRE+PHOTO)
## HASSI ASMAE CAC 2 
#  Prédiction du Diabète - Projet Machine Learning

##  Table des Matières
- [ Aperçu du Projet](#-aperçu-du-projet)
- [ Dataset](#-dataset)
- [ Méthodologie](-méthodologie)
- [ Résultats](#-résultats)
- [ Installation et Usage](#-installation-et-usage)
- [ Contribution](#-contribution)
- [ License](#-license)

---

##  Aperçu du Projet

###  Description
Ce projet vise à développer un modèle de machine learning pour prédire le risque de diabète chez les patientes à partir de données médicales simples.

###  Objectifs
-  Analyser les facteurs de risque du diabète
-  Développer un modèle prédictif performant
-  Identifier les variables les plus importantes
-  Proposer une application clinique potentielle

###  Métriques Clés
| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 85.2% |
| **F1-Score** | 81.2% |
| **AUC-ROC** | 0.892 |
| **Meilleur modèle** | Random Forest |

---

##  Dataset

###  Informations Générales
- **Nom**: Pima Indians Diabetes Database
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Instances**: 768 patientes
- **Features**: 8 caractéristiques médicales
- **Target**: Diabétique (1) / Non-diabétique (0)

###  Variables du Dataset

| Variable | Type | Description | Unité |
|----------|------|-------------|-------|
| `Pregnancies` | Numérique | Nombre de grossesses | Nombre |
| `Glucose` | Numérique | Concentration glucose | mg/dL |
| `BloodPressure` | Numérique | Pression artérielle | mm Hg |
| `SkinThickness` | Numérique | Épaisseur peau triceps | mm |
| `Insulin` | Numérique | Insulinémie 2h | mu U/ml |
| `BMI` | Numérique | Indice masse corporelle | kg/m² |
| `DiabetesPedigreeFunction` | Numérique | Risque génétique | Score |
| `Age` | Numérique | Âge patiente | Années |
| `Outcome` | Catégoriel | Variable cible | Classe |


## Méthodologie

### 1. Analyse Exploratoire des Données
- Analyse des valeurs manquantes et des valeurs zero
- Statistiques descriptives
- Analyse des corrélations
- Visualisation des distributions

### 2. Prétraitement des Données
- Gestion des valeurs manquantes (remplacement par la médiane)
- Normalisation des features
- Split des données (80% train, 20% test)

### 3. Modèles de Machine Learning Implémentés
- Random Forest Classifier
- Logistic Regression
- Gradient Boosting Classifier
- Support Vector Machine
  
### DESCRIPTION DU CODE ET INTERPRÉTATION DES RÉSULTATS

## ÉTAPE 1: CHARGEMENT DES PACKAGES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

Description: Cette étape importe toutes les bibliothèques nécessaires pour le projet. Pandas et NumPy sont utilisés pour la manipulation des données, Matplotlib et Seaborn pour la visualisation, et Scikit-learn pour les algorithmes de machine learning et l'évaluation des modèles.

ÉTAPE 2: TÉLÉCHARGEMENT DU DATASET
python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)
Description: Télécharge le dataset depuis GitHub et attribue des noms explicites aux colonnes pour une meilleure lisibilité. Le dataset contient des données médicales de patients avec diabète.

ÉTAPE 3: VÉRIFICATION DES DONNÉES
python
print(df.head())  # Aperçu des premières lignes
print(df.info())  # Informations sur les types de données
print(df.describe())  # Statistiques descriptives
print(df['Outcome'].value_counts())  # Distribution de la variable cible
Résultats et Interprétation:

Dimensions du dataset: 768 lignes × 9 colonnes

Distribution de la variable cible:

0 (Non diabétique): 500 cas (65.1%)

1 (Diabétique): 268 cas (34.9%)

Taux de prévalence: 34.9% - indique un déséquilibre modéré des classes

Analyse statistique initiale:

L'âge moyen des patients est de 33.2 ans

Le glucose moyen est de 120.9 mg/dL

L'IMC moyen est de 32.0 kg/m² (surpoids)

ÉTAPE 4: NETTOYAGE ET PRÉPARATION
python
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    df_clean[col] = df_clean[col].replace(0, df_clean[col].median())
Problème identifié: Présence de valeurs zero biologiquement impossibles dans les mesures médicales.

Traitement appliqué:

Glucose: 5 valeurs zero remplacées (0.65%)

BloodPressure: 35 valeurs zero remplacées (4.56%)

SkinThickness: 227 valeurs zero remplacées (29.56%)

Insulin: 374 valeurs zero remplacées (48.70%)

BMI: 11 valeurs zero remplacées (1.43%)

Justification: Les valeurs zero dans ces mesures médicales représentent des valeurs manquantes. Le remplacement par la médiane préserve la distribution des données tout en permettant l'analyse.

**ÉT



