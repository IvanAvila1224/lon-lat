import streamlit as st
import requests

SERVER_URL = 'https://map-model-service-icec-yoangelcruz.cloud.okteto.net/v1/models/map-model:predict'

def make_prediction(inputs):
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    
    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        st.error("Failed to get predictions. Please check your inputs and try again.")
        return None

def main():
    st.title('Predictor de ubicaciones geográficas')

    st.header('Coordenadas para Kazajistán')
    kazakhstan_lat = st.number_input('Ingrese la latitud de Kazajistán:', value=48.0196)
    kazakhstan_lon = st.number_input('Ingrese la longitud de Kazajistán:', value=66.9237)

    st.header('Coordenadas para Brasilia')
    brasilia_lat = st.number_input('Ingrese la latitud de Brasilia:', value=-15.7801)
    brasilia_lon = st.number_input('Ingrese la longitud de Brasilia:', value=-47.9292)

    if st.button('Predecir'):
        inputs = [
            [kazakhstan_lon, kazakhstan_lat],
            [brasilia_lon, brasilia_lat]
        ]
        predictions = make_prediction(inputs)

        if predictions:
            st.write("\nPredicciones para Kazajistán:")
            st.write(predictions['predictions'][0])

            st.write("\nPredicciones para Brasilia:")
            st.write(predictions['predictions'][1])

if __name__ == '__main__':
    main()