import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras import layers, models

# Charger les données depuis un fichier CSV
data = pd.read_csv('C:\\Users\\moi\\Documents\\myfile.csv')  # Remplace par le chemin de ton fichier CSV

# Afficher les noms de colonnes pour identifier la colonne cible
print("Noms de colonnes :", data.columns)

# Supposons que nous définissons la colonne cible ici
data['target'] = data.apply(lambda row: 1 if 'malicious_condition' in row['Info'] else 0, axis=1)

# Séparer les features (X) et les labels (y)
X = data.drop(columns=['No.', 'Time', 'Info', 'target'])  # Exclure les colonnes inutiles
y = data['target']

# Déterminer les colonnes numériques et catégorielles
num_cols = ['Length']  # Ajouter d'autres colonnes numériques si nécessaire
cat_cols = ['Source', 'Destination', 'Protocol']  # Les colonnes à encoder

# Créer un prépro    cesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ])

# Créer un pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Appliquer le préprocesseur aux données
X_processed = pipeline.fit_transform(X)

# Diviser les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Créer le modèle
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Pour la classification binaire

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Précision du modèle : {accuracy * 100:.2f}%')
