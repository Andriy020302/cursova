from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Завантаження моделі
model = joblib.load('model/computer_configuration_model.pkl')

# Завантаження даних
cpu_data = pd.read_csv('data/Intel_CPUs.csv')
gpu_data = pd.read_csv('data/All_GPUs.csv')

# Функції для обробки даних
def convert_price(price):
    if '-' in price:
        low, high = price.split('-')
        return (float(low) + float(high)) / 2
    else:
        return float(price)

def convert_frequency(freq):
    if isinstance(freq, str):
        if 'GHz' in freq:
            return float(freq.replace(' GHz', ''))
        elif 'MHz' in freq:
            return float(freq.replace(' MHz', '')) / 1000
    return freq

def convert_memory(memory):
    if isinstance(memory, str):
        if 'GB' in memory:
            return float(memory.replace(' GB', ''))
        elif 'MB' in memory:
            return float(memory.replace(' MB', '')) / 1024
    return memory

def convert_core_speed(speed):
    if isinstance(speed, str):
        speed = speed.strip()  # Видалення зайвих пробілів та символів
        if ' MHz' in speed:
            return float(speed.replace(' MHz', ''))
        elif speed == '-' or speed == '':
            return np.nan
    return speed

# Очищення даних
cpu_data['Recommended_Customer_Price'] = cpu_data['Recommended_Customer_Price'].str.replace('[\$,]', '', regex=True).str.strip()
cpu_data['Recommended_Customer_Price'] = cpu_data['Recommended_Customer_Price'].replace('', pd.NA).dropna().apply(convert_price)
cpu_data['Processor_Base_Frequency'] = cpu_data['Processor_Base_Frequency'].apply(convert_frequency)
gpu_data['Memory'] = gpu_data['Memory'].apply(convert_memory)
gpu_data['Core_Speed'] = gpu_data['Core_Speed'].apply(convert_core_speed).dropna().astype(float)
cpu_data['nb_of_Cores'] = cpu_data['nb_of_Cores'].astype(float)
cpu_data['nb_of_Threads'] = cpu_data['nb_of_Threads'].astype(float)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    max_price = float(request.form['max_price'])
    min_cpu_cores = float(request.form['min_cpu_cores'])
    min_cpu_frequency = float(request.form['min_cpu_frequency'])
    min_gpu_memory = float(request.form['min_gpu_memory'])
    min_gpu_core_speed = float(request.form['min_gpu_core_speed'])

    cpu_features = ['Recommended_Customer_Price', 'Processor_Base_Frequency', 'nb_of_Cores', 'nb_of_Threads']
    gpu_features = ['Memory', 'Core_Speed']

    cpu_data_filtered = cpu_data[cpu_features].dropna()
    gpu_data_filtered = gpu_data[gpu_features].dropna()

    cpu_data_filtered = cpu_data_filtered.apply(pd.to_numeric, errors='coerce').dropna()
    gpu_data_filtered = gpu_data_filtered.apply(pd.to_numeric, errors='coerce').dropna()

    filtered_cpus = cpu_data_filtered[(cpu_data_filtered['Recommended_Customer_Price'] <= max_price) &
                                      (cpu_data_filtered['nb_of_Cores'] >= min_cpu_cores) &
                                      (cpu_data_filtered['Processor_Base_Frequency'] >= min_cpu_frequency)]

    filtered_gpus = gpu_data_filtered[(gpu_data_filtered['Memory'] >= min_gpu_memory) &
                                      (gpu_data_filtered['Core_Speed'] >= min_gpu_core_speed)]

    min_length = min(len(filtered_cpus), len(filtered_gpus))
    if min_length == 0:
        return render_template('result.html', error="Немає конфігурацій, які відповідають заданим критеріям.")

    filtered_cpus = filtered_cpus.iloc[:min_length]
    filtered_gpus = filtered_gpus.iloc[:min_length]

    combined_data = pd.concat([filtered_cpus.reset_index(drop=True), filtered_gpus.reset_index(drop=True)], axis=1)

    combined_data['Predicted_Performance'] = model.predict(combined_data[cpu_features + gpu_features])

    top_configurations = combined_data.sort_values(by='Predicted_Performance', ascending=False).head(5)

    return render_template('result.html', configurations=top_configurations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
