import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import shap


def discriminative(X, y, feature_names):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)

   print(X_train.shape[1])
   model = keras.Sequential([
     keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
     keras.layers.Dense(1, activation='sigmoid')
   ])

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1, validation_data=(X_test, y_test))


   # Use SHAP to explain feature importance
   explainer = shap.Explainer(model, X_train)
   shap_values = explainer(X_test)
   #print(shap_values.values)
   shap_importance = np.abs(shap_values.values).mean(axis=0)
   feature_ranking = pd.DataFrame({'Feature': feature_names, 'SHAP_Importance': shap_importance})
   feature_ranking = feature_ranking.sort_values(by="SHAP_Importance", ascending=False)

   blocking_keys = feature_ranking.iloc[:]["Feature"].tolist()

   # Visualize feature importance
   #shap.summary_plot(shap_values, X_test, feature_names=feature_names)
   return blocking_keys, shap_importance
