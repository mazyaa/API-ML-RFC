from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)
load_dotenv()

DATABASE = os.getenv('DATABASE')
MODEL = os.getenv('MODEL')


#load model
model = joblib.load(MODEL)

#start date for prediction
start_date = datetime.strptime('2021-08-01', '%Y-%m-%d')

# function to decode tanggal
def decode_tanggal(days_since_start):
    target_date = start_date + timedelta(days=int(days_since_start))
    return target_date.strftime('%Y-%m-%d')

#decoded nama_produk
nama_produk_decoded = {
    0: 'Bolen Banana',
    1: 'Bolen Cokju (Mini)',
    2: 'Bolen Coklat',
    3: 'Bolen Coklat Keju',
    4: 'Bolen Keju Mini',
    5: 'Bolen Pisang Coklat',
    6: 'Bolen Proltape',
}

#decoded status prediksi (status stok)
status_decoded = {
    0: 'normal',
    1: 'overstock',
    2: 'understock'
}

#mapping hari
hari_mapping = {
    1: 'Senin',
    2: 'Selasa',
    3: 'Rabu',
    4: 'Kamis',
    5: 'Jumat',
    6: 'Sabtu',
    7: 'Minggu'
}

#save to database
def save_to_db(data, prediction):
    conn = sqlite3.connect(DATABASE)  # Connect to the SQLite database
    cursor = conn.cursor()  # Create a cursor object
    
    #mapping and decode tanggal,  nama produk, status prrediksi and hari to string
    tanggal_asli = decode_tanggal(data['tanggal'])
    nama_produk_asli = nama_produk_decoded.get(data['nama_produk'], 'Unknown')
    status_asli = status_decoded.get(prediction, str(prediction))
    hari_asli = hari_mapping.get(data['hari'], 'Unknown')
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT,
            nama_produk INTEGER,
            hari INTEGER,
            stok_produk INTEGER,
            harga_satuan INTEGER,
            jumlah_terjual INTEGER,
            status_prediksi TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO predictions (tanggal, nama_produk, hari, stok_produk, harga_satuan, jumlah_terjual, status_prediksi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        tanggal_asli,
        nama_produk_asli,
        hari_asli,
        data['stok_produk'],
        data['harga_satuan'],
        data['jumlah_terjual'],
        status_asli 
    ))
    conn.commit()  # Commit the changes
    conn.close()  # Close the connection

#endpoint to predict
@app.route('/predict', methods=['POST'])
def prediction_sales():
    try:
        data = request.get_json()
        input_array = np.array([[
            data['tanggal'],
            data['nama_produk'],
            data['hari'],
            data['stok_produk'],
            data['harga_satuan'],
            data['jumlah_terjual']
        ]], dtype=object)

        
        prediction = model.predict(input_array)[0] # Predict the class label
        probabilities = model.predict_proba(input_array)[0]  # Get the probabilities for each class
        confidence = np.max(probabilities)  # Get the maximum probability as confidence
        
        if prediction == 0:
            prediction = 'normal'
        elif prediction == 1:
            prediction = 'overstock'
        elif prediction == 2:
            prediction = 'understock'
            
        save_to_db(data, str(prediction))
        
        # Return the prediction result
        return jsonify({
            'status': 200,
            'message': 'Prediction successful',
            'prediction': str(prediction),
            'confidence' : round(confidence * 100, 2), # Confidence percentage
        })
    except Exception as e:
        return jsonify({
            'status': 500,
            'message': 'Error occurred during prediction',
            'error': str(e)
        })
        
#endpoint to get all predictions
@app.route('/get-all-predictions', methods=['GET'])
def get_all_data():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions')
    rows = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'status': 200,
        'message': 'Data retrieved successfully',
        'data': [
            {
                'id': row[0],
                'tanggal': row[1],
                'nama_produk': row[2],
                'hari': row[3],
                'stok_produk': row[4],
                'harga_satuan': row[5],
                'jumlah_terjual': row[6],
                'status_prediksi': row[7]
            } for row in rows
        ]
    })

#endpoint import csv to db
@app.route('/import-csv', methods=['POST'])
def import_csv():
    try:
        data = request.get_json()
        csv_path = data.get('csv_path')
        df = pd.read_csv(csv_path)
        
        conn = sqlite3.connect(DATABASE)
        df.to_sql('predictions', conn, if_exists='replace', index=False)
        conn.close()
        
        return jsonify({
            'status': 200,
            'message': 'CSV imported successfully',
        })
        
    except Exception as e:
        return jsonify({
            'status': 500,
            'message': 'Error occurred during CSV import',
            'error': str(e)
        })
        
#endpoint to delete all data
@app.route('/delete-all-data', methods=['DELETE'])
def delete_all_data():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 200,
            'message': 'All data deleted successfully',
        })
        
    except Exception as e:
        return jsonify({
            'status': 500,
            'message': 'Error occurred during deletion',
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)

