import os
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIR_REPO = Path.cwd().parent.parent
DIR_DATA_PROCESSED = Path(DIR_REPO) / "data" / "processed"
DIR_MODELS = Path(DIR_REPO) / "models"

# CARGA DE DATOS PROCESADOS
def load_processed_data(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Datos cargados correctamente desde {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        raise

# PREPROCESAMIENTO DEL CONJUNTO DE DATOS
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
    MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}

    df['neighbourhood'] = df['neighbourhood'].map(MAP_NEIGHB)
    df['room_type'] = df['room_type'].map(MAP_ROOM_TYPE)
    
    df = df.dropna(axis=0)
    
    return df

# CLASIFICADOR RANDOM FOREST
def train_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced', n_jobs=4)
    clf.fit(X_train, y_train)
    logger.info("Finished training")
    return clf

# EVALUACIÓN DEL MODELO 
def evaluate_model(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_proba = clf.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    return y_pred

# VISUALIZACIÓN DE LA IMPORTANCIA DE LAS CARACTERÍSTICAS
def plot_feature_importance(clf: RandomForestClassifier, feature_names: list):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = np.array(feature_names)[indices]
    importances = importances[indices]

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), features, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Features importance", fontsize=12)
    plt.show()

# VISUALIZACIÓN DE LA MATRIZ DE CONFUSIÓN
def plot_confusion_matrix(y_test, y_pred, classes: list, labels: list):
    c = confusion_matrix(y_test, y_pred)
    c = c / c.sum(axis=1).reshape(len(classes), 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(c, annot=True, cmap='BuGn', square=True, fmt='.2f', annot_kws={'size': 10}, cbar=False)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Real', fontsize=16)
    plt.xticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
    plt.yticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
    plt.title("Confussion Matrix", fontsize=18)
    plt.show()

# VISUALIZACIÓN DEL INFORME DE CLASIFICACIÓN
def plot_classification_report(y_test, y_pred, labels: list):
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame.from_dict(report).T[:-3]
    df_report.index = labels

    metrics = ['precision', 'recall', 'support']
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 7))

    for i, ax in enumerate(axes):
        ax.barh(df_report.index, df_report[metrics[i]], alpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel(metrics[i], fontsize=12)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Classification Report", fontsize=14)
    plt.show()

# GUARDAR EL MODELO
def save_model(clf: RandomForestClassifier, model_path: Path):
    with open(model_path, 'wb') as model_file:
        pickle.dump(clf, model_file)
    logger.info(f"Model saved in {model_path}")

# FUNCIÓN MAIN
def main():
    # CARGAR LOS DATOS
    filepath_processed = DIR_DATA_PROCESSED / "preprocessed_listings.csv"
    df = load_processed_data(filepath_processed)
    
    # PREPROCESAMIENTO
    df = preprocess_data(df)
    
    # CARACTERÍSTICAS (X) Y OBJETIVO (y)
    FEATURE_NAMES = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']
    X = df[FEATURE_NAMES]
    y = df['category']
    
    # DIVISIÓN EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    
    # ENTRENAR CLASIFICADOR
    clf = train_classifier(X_train, y_train)
    
    # EVALUAR MODELO
    y_pred = evaluate_model(clf, X_test, y_test)
    
    # RESULTADOS
    plot_feature_importance(clf, FEATURE_NAMES)
    plot_confusion_matrix(y_test, y_pred, classes=[0, 1, 2, 3], labels = ['low', 'mid', 'high', 'lux'])
    plot_classification_report(y_test, y_pred, labels = ['low', 'mid', 'high', 'lux'])
    
    # GUARDAR MODELO
    model_path = DIR_MODELS / "simple_classifier.pkl"
    save_model(clf, model_path)

if __name__ == "__main__":
    main()
