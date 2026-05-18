# Prédiction du Churn Client — Machine Learning & Deep Learning

## Description
Ce projet vise à prédire le churn client en exploitant deux approches complémentaires :
- **Partie 1 — Machine Learning** : données tabulaires (Telco Customer Churn Dataset)
- **Partie 2 — Deep Learning** : données textuelles (Customer Support on Twitter Dataset)

## Datasets
- Telco Customer Churn : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Customer Support on Twitter : https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

## Structure du projet
churn-prediction/
├── Partie1_ML_Churn.ipynb       # Machine Learning sur données tabulaires
├── Partie2_DL_Churn.ipynb       # Deep Learning sur données textuelles
├── requirements.txt              # Dépendances Python
└── README.md

## Résultats

### Partie 1 — Machine Learning
| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.7410 | 0.5076 | 0.7995 | 0.6210 |
| Decision Tree | 0.7630 | 0.5446 | 0.6524 | 0.5937 |
| Random Forest | 0.7729 | 0.5754 | 0.5508 | 0.5628 |
| Gradient Boosting | 0.7842 | 0.5941 | 0.5909 | 0.5925 |
| SVM | 0.7573 | 0.5316 | 0.7193 | 0.6114 |

### Partie 2 — Deep Learning
| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression | 0.9048 | 0.7912 | 0.8392 | 0.8145 |
| Word2Vec + Logistic Regression | 0.6659 | 0.3981 | 0.6673 | 0.4986 |
| Bi-LSTM + Word2Vec | 0.9145 | 0.8232 | 0.8365 | 0.8298 |
| DistilBERT | 0.9293 | 0.8719 | 0.8392 | 0.8552 |

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation

### Partie 1 — Machine Learning (local)
Télécharger le dataset Telco et placer le fichier `WA_Fn-UseC_-Telco-Customer-Churn.csv` dans le même dossier que le notebook.

### Partie 2 — Deep Learning (Google Colab recommandé si pas de GPU sur l'ordinateur)
La partie 2 a été réalisée sur Google Colab avec GPU T4 en raison des ressources computationnelles requises par les modèles LSTM et DistilBERT. Il est recommandé de l'exécuter sur Colab en montant le dataset `twcs.csv` depuis Google Drive :
```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/twcs.csv')
```

## Environnement
- Python 3.10
- Google Colab (GPU T4) pour la Partie 2
- VS Code pour la Partie 1