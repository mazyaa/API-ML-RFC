from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta


app = Flask(__name__)
CORS(app)

load_dotenv()
DATABASE = os.getenv('DATABASE')
MODEL = os.getenv('MODEL')


#for load model
model = joblib.load('model_rfc_final.pkl')

#start date for prediction
start_date = datetime.strptime('2021-08-01', '%Y-%m-%d')

#function to decode tanggal
def decode_tanggal(days_since_start):
    target_date = start_date + timedelta(days=int(days_since_start))
    return target_date.strftime('%Y-%m-%d')

#for decoded nama_produk
nama_produk_decoded = {
    0: 'Bolen Banana',
    1: 'Bolen Cokju (Mini)',
    2: 'Bolen Coklat',
    3: 'Bolen Coklat Keju',
    4: 'Bolen Keju Mini',
    5: 'Bolen Pisang Coklat',
    6: 'Bolen Proltape',
}

#for decoded (status stok)
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
    tanggal_asli = (start_date + timedelta(days=int(data['tanggal']))).strftime('%m-%d-%y')
    nama_produk_asli = nama_produk_decoded.get(data['nama_produk'], 'Unknown')
    status_asli = status_decoded.get(prediction, str(prediction))
    hari_asli = hari_mapping.get(data['hari'], 'Unknown')
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT,
            hari INTEGER,
            nama_produk TEXT,
            harga_satuan INTEGER,
            jumlah_terjual INTEGER,
            stok_produk INTEGER,
            status_stok TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO predictions (tanggal, hari, nama_produk, harga_satuan, jumlah_terjual, stok_produk, status_stok)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        tanggal_asli,
        hari_asli,
        nama_produk_asli,
        data['harga_satuan'],
        data['jumlah_terjual'],
        data['stok_produk'],
        status_asli
    ))
    conn.commit()  # Commit the changes
    conn.close()  # Close the connection
    
#create table
@app.route('/create-table', methods=['POST'])
def create_table():
    try:
        conn = sqlite3.connect(DATABASE)  # Connect to the SQLite database
        cursor = conn.cursor()  # Create a cursor object
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tanggal TEXT,
                hari INTEGER,
                nama_produk TEXT,
                harga_satuan INTEGER,
                jumlah_terjual INTEGER,
                stok_produk INTEGER,
                status_stok TEXT
            )
        ''')
        conn.commit()  # Commit the changes
        conn.close()  # Close the connection
        return jsonify({
            'status': 200,
            'message': 'Table created successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 500,
            'message': 'Error occurred while creating table',
            'error': str(e)
        })

#endpoint to predict
@app.route('/predict', methods=['POST'])
def prediction_sales():
    try:
        data = request.get_json()
      
        input_array = np.array([[
            data['tanggal'],
            data['hari'],
            data['nama_produk'],
            data['harga_satuan'],
            data['stok_produk'],
            data['jumlah_terjual'],
        ]], dtype=object)

        
        prediction = model.predict(input_array)[0] # Predict the class label
        probabilities = model.predict_proba(input_array)[0]  # Get the probabilities for each class
        confidence = np.max(probabilities)  # Get the maximum probability as confidence
        
        with open('model_evaluation.json', 'r') as f:
            model_evaluation = json.load(f)
            
        model_accuracy = model_evaluation['Accuracy']
        model_precision = model_evaluation['Precision']
        model_recall = model_evaluation['Recall']
        model_f1_score = model_evaluation['F1_Score']
        
        prediction_encoded = prediction
        
        if prediction == 0:
            prediction = 'normal'
        elif prediction == 1:
            prediction = 'overstock'
        elif prediction == 2:
            prediction = 'understock'
            
        save_to_db(data, prediction_encoded)
        
        # Return the prediction result
        return jsonify({
            'status': 200,
            'message': 'Prediction successful',
            'prediction': str(prediction),
            'accuracy' : round(model_accuracy * 100, 2), 
            'precision' : round(model_precision * 100, 2), 
            'recall' : round(model_recall * 100, 2), 
            'F1_score' : round(model_f1_score * 100, 2), 
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
    cursor.execute('SELECT id, tanggal, hari, nama_produk, harga_satuan, jumlah_terjual, stok_produk, status_stok FROM predictions ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'status': 200,
        'message': 'Data retrieved successfully',
        'data': [
            {
                'id': row[0],
                'tanggal': row[1],
                'hari': row[2],
                'nama_produk': row[3],
                'harga_satuan': row[4],
                'jumlah_terjual': row[5],
                'stok_produk': row[6],
                'status_stok': row[7],
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
        
        #before importing, create col id
        df.insert(0, 'id', range(1, len(df)+ 1))
        
        # DECODE Hari
        df['hari'] = df['hari'].map(hari_mapping)
        
        conn = sqlite3.connect(DATABASE)
        df.to_sql('predictions', conn, if_exists='append', index=False)
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

