# Flask API - Prediksi Status Stok Penjualan Bolen Crispy

Project ini adalah API Machine Learning berbasis Flask untuk memprediksi status stok produk Bolen Crispy (Understock, Normal, Overstock) menggunakan model Random Forest Classifier.

## Fitur API:

- ✅ POST /predict → Melakukan prediksi berdasarkan data input
- ✅ GET /data → Mengambil semua histori prediksi dari database SQLite
- ✅ POST /import_csv → Import dataset CSV ke database (opsional)

## Cara Menjalankan

1. Buat dan aktifkan virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows





