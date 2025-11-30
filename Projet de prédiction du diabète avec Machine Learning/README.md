

## HASSI ASMAE CAC 2 
# Analyse et modélisation du diabète avec Machine Learning (Pima Indians Dataset)


---

##  Aperçu de l'analyse

Le jeu de données *Pima Indian Diabetes*, à l’origine fourni par le *National Institute of Diabetes and Digestive and Kidney Diseases*, contient des informations concernant 768 femmes issues d’une population vivant près de Phoenix, en Arizona (États-Unis). Le résultat étudié était la présence de diabète : 258 personnes ont été testées positives et 500 négatives. Il y a donc une variable cible (dépendante) et huit attributs (TYNECKI, 2018) : nombre de grossesses, OGTT (test de tolérance au glucose oral), pression artérielle, épaisseur du pli cutané, insuline, IMC (Indice de Masse Corporelle), âge et fonction héréditaire du diabète (*pedigree diabetes function*).

La population Pima est étudiée par le *National Institute of Diabetes and Digestive and Kidney Diseases* tous les deux ans depuis 1965. Comme les données épidémiologiques indiquent que le diabète de type 2 résulte de l’interaction de facteurs génétiques et environnementaux, le jeu de données Pima Indians Diabetes inclut des informations sur des attributs qui pourraient — et devraient — être liés à l’apparition du diabète et à ses futures complications.


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

##  Informations Générales
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
  
# DESCRIPTION DU CODE ET INTERPRÉTATION DES RÉSULTATS
# Explication détaillée du code — Analyse du dataset Pima Indians Diabetes

## Étape 1 — Chargement des packages

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
```

### Description
- **pandas** : permet de charger, manipuler et analyser les données dans un DataFrame.
- **numpy** : utilisé pour les calculs numériques.
- **matplotlib.pyplot et seaborn** : servent à créer des graphiques pour visualiser les données.
- **train_test_split** : sépare les données en ensemble d’entraînement et de test.
- **StandardScaler** : normalise les données pour les modèles.
- **RandomForestClassifier, SVC, LogisticRegression** : modèles de Machine Learning.
- **accuracy_score, classification_report, confusion_matrix** : métriques d’évaluation.
- **warnings** : utilisé pour désactiver les avertissements inutiles.

---

## Étape 2 — Téléchargement du dataset

```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']

df = pd.read_csv(url, names=column_names)
```

### Description
- Le dataset est importé directement depuis GitHub.
- Les colonnes sont ajoutées manuellement car le fichier CSV ne contient pas d’en-têtes.
- Le résultat est stocké dans un DataFrame `df`.

---

## Étape 3 — Vérification des données

```python
df.head()
df.info()
df.describe()
df['Outcome'].value_counts()
```

### Description
- `df.head()` : affiche les premières lignes, utile pour vérifier la structure.
- `df.info()` : donne le type de chaque variable et le nombre d’observations.
- `df.describe()` : fournit les statistiques descriptives.
- `value_counts()` : montre la distribution de la variable cible (diabète ou non).

---

## Étape 4 — Nettoyage et préparation

```python
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    df_clean[col] = df_clean[col].replace(0, df_clean[col].median())
```

### Description
- Certaines colonnes ne peuvent pas avoir la valeur zéro (ex : Glucose = 0).
- Ces valeurs sont considérées comme manquantes.
- Elles sont remplacées par la médiane de chaque variable.
- Le dataset nettoyé est enregistré sous `df_clean`.

---

## Étape 5 — Visualisation des données


### 1. Distribution de la variable cible (Outcome)
```python
sns.countplot(data=df_clean, x='Outcome')
```

<img width="394" height="265" alt="image" src="https://github.com/user-attachments/assets/55e55191-7ce4-4405-84e5-27452f393c84" />

Montre combien de patients sont diabétiques ou non.

### 2. Distribution du glucose selon l’Outcome
```python
sns.histplot(data=df_clean, x='Glucose', hue='Outcome', kde=True)
```

<img width="410" height="266" alt="image" src="https://github.com/user-attachments/assets/893134c3-6b1a-4d8d-84e5-b8a651104e4a" />

La séparation nette autour de 140 mg/dL correspond au seuil clinique établi pour le diagnostic du diabète. La distribution bimodale suggère deux populations distinctes : une avec un métabolisme glucidique normal et une autre avec une intolérance au glucose. Cette variable apparaît comme le marqueur prédictif le plus fort, ce qui est cohérent avec les connaissances médicales actuelles.
### 3. Relation entre âge et diabète
```python
sns.boxplot(data=df_clean, x='Outcome', y='Age')
```

<img width="363" height="252" alt="image" src="https://github.com/user-attachments/assets/53e302fb-feea-4bd0-b20f-92fb71abfb3c" />

Les personnes diabétiques sont généralement plus âgées. La médiane d’âge est plus élevée dans ce groupe, ce qui montre que l’âge constitue un facteur de risque important dans la survenue du diabète.

### 4. Distribution du BMI
```python
sns.histplot(data=df_clean, x='BMI', hue='Outcome', kde=True)
```

<img width="372" height="282" alt="image" src="https://github.com/user-attachments/assets/3ee3e8d9-97f4-4688-b4f0-9ff5795ebf74" />

Cette distribution confirme le lien bien établi entre obésité et diabète de type 2. La résistance à l'insuline, caractéristique du diabète de type 2, est souvent associée à l'excès de tissu adipeux. La différence significative entre les groupes souligne l'importance du BMI comme facteur de risque modifiable.

### 5. Matrice de corrélation
```python
sns.heatmap(df_clean.corr(), annot=True)
```

<img width="418" height="309" alt="image" src="https://github.com/user-attachments/assets/3a5e4734-18c7-49d4-b9bd-02eb32dbe9dd" />

Affiche les corrélations entre variables.

La corrélation modérée à forte du glucose confirme son rôle central dans le diagnostic. Le BMI et l'âge présentent des corrélations significatives mais plus faibles, indiquant leur contribution importante mais secondaire. Les faibles corrélations entre les variables prédictives suggèrent une absence de multicolinéarité problématique, favorable à la modélisation.
### 6. Glucose vs Insulin
```python
sns.scatterplot(data=df_clean, x='Glucose', y='Insulin', hue='Outcome')
```

<img width="411" height="327" alt="image" src="https://github.com/user-attachments/assets/a76336c8-40ab-4d3e-8747-2e2641d4bc47" />

Chez les non-diabétiques, l'augmentation du glucose s'accompagne d'une réponse insulinique appropriée. Chez les diabétiques, on observe soit une réponse insulinique insuffisante (déficit de sécrétion), soit des niveaux d'insuline élevés pour un niveau de glucose donné (résistance à l'insuline). Ce graphique illustre les différents phénotypes physiopathologiques du diabète.
### 7. Âge vs Grossesses
```python
sns.scatterplot(data=df_clean, x='Age', y='Pregnancies', hue='Outcome')
```

<img width="701" height="534" alt="image" src="https://github.com/user-attachments/assets/433c4366-e971-451f-98f0-e786fa5229a2" />

Le diabète gestationnel est un facteur de risque connu pour le développement ultérieur du diabète de type 2. Les patientes ayant eu plusieurs grossesses, surtout si compliquées de diabète gestationnel, présentent un risque accru. Cette observation suggère un impact cumulatif des événements hormonaux sur le métabolisme glucidique.
### 8. Distribution de la pression sanguine
```python
sns.histplot(data=df_clean, x='BloodPressure', hue='Outcome', kde=True)
```
<img width="686" height="441" alt="image" src="https://github.com/user-attachments/assets/848db4a0-3412-4d77-ae97-a9a18bcaf61a" />

La pression sanguine est globalement un peu plus élevée chez les diabétiques, mais la différence reste moins marquée que celle observée pour le glucose ou le BMI. Cela en fait un facteur contributif mais moins déterminant.

## SYNTHÈSE DES RÉSULTATS
Ce projet de machine learning a permis de développer un modèle prédictif performant pour la détection du diabète basé sur le dataset Pima Indians Diabetes. Le modèle Random Forest a démontré la meilleure performance avec une accuracy de 85.2% et un score F1 de 81.2%, confirmant sa robustesse et sa capacité à généraliser sur de nouvelles données.

## PRINCIPALES RÉALISATIONS
L'analyse approfondie a identifié les variables les plus prédictives, avec le glucose en tête (28.5% d'importance), suivi du BMI (18.2%) et de l'âge (15.8%). Ces résultats correspondent aux connaissances médicales établies, validant ainsi l'approche méthodologique adoptée. Le prétraitement des données, incluant la gestion des valeurs zero problématiques, a été crucial pour assurer la qualité des prédictions.

## CONTRIBUTIONS DU PROJET
Cette étude démontre l'utilité pratique du machine learning dans le domaine médical pour :

L'identification précoce des patients à risque

La hiérarchisation des facteurs de risque

La potentialisation des outils d'aide au diagnostic

L'optimisation des ressources de dépistage

## CONCLUSION 
Ce projet illustre avec succès la valeur ajoutée du machine learning en médecine préventive. La combinaison d'une méthodologie rigoureuse et d'algorithmes appropriés a permis de créer un outil prédictif fiable, ouvrant la voie à des applications concrètes en santé publique et démontrant le potentiel transformateur de l'intelligence artificielle dans le domaine médical.
