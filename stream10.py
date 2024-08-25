import streamlit as st
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un dataset de LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Configuración básica del modelo LightGBM
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Callback para early stopping
callbacks = [
    lgb.early_stopping(stopping_rounds=10)
]

# Entrenar el modelo
bst = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, callbacks=callbacks)

# Interfaz de Streamlit
#st.title("Predicción de Iris con LightGBM")

# Entrada de datos del usuario
#st.header("Introduce las características de la flor:")

st.markdown("<h1 style='font-size: 36px;'>Predicción de Iris con LightGBM</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 24px;'>Introduce las características de la flor:</h2>", unsafe_allow_html=True)

input_data = []
for i, feature_name in enumerate(feature_names):
    val = st.slider(f"{feature_name}",
                    float(X[:, i].min()),  # Rango mínimo para esta característica
                    float(X[:, i].max()),  # Rango máximo para esta característica
                    float(X[:, i].mean()))  # Valor por defecto (media)
    input_data.append(val)

# Realizar la predicción
input_data = np.array(input_data).reshape(1, -1)
prediction = bst.predict(input_data)
predicted_class = np.argmax(prediction)

# Mostrar resultados
#st.write(f"Clase predicha: **{target_names[predicted_class]}**")
st.markdown(f"Clase predicha: **<span style='color:red;'>{target_names[predicted_class]}</span>**", unsafe_allow_html=True)

st.write("Probabilidades:")
for i, prob in enumerate(prediction[0]):
    st.write(f"{target_names[i]}: {prob:.2f}")


