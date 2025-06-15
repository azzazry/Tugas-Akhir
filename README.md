# Deteksi Insider Threat Menggunakan Graph Neural Network (GraphSAGE) dan Interpretasi GraphSVX
Proyek ini merupakan bagian dari tugas akhir sarjana pada bidang Teknologi Informasi. Fokus utama riset adalah pada deteksi anomali perilaku karyawan di lingkungan korporat (_insider threat_) yang sulit dikenali dengan metode konvensional. Implementasi ini menggabungkan _Graph Neural Network_ (GNN) dan metode interpretasi berbasis GraphSVX untuk meningkatkan akurasi dan transparansi dalam proses deteksi. Inspirasi awal berasal dari repositori [GraphSVX](https://github.com/AlexDuvalinho/GraphSVX), yang kemudian dimodifikasi dan dikembangkan lebih lanjut sesuai konteks keamanan siber.

### Abstrak
Anomali dalam sistem komputer, seperti perilaku _insider threat_, merupakan ancaman serius yang kerap luput dari pengawasan karena menyamar sebagai aktivitas normal. _Graph Neural Network_ (GNN), khususnya arsitektur GraphSAGE, menawarkan pendekatan representasi relasional yang efektif untuk mengungkap pola tersembunyi dalam data berbasis graf. Dalam proyek ini, dibangun sistem klasifikasi biner pada graf heterogen yang terdiri dari node bertipe `user`, `pc`, dan `url`. Untuk menjamin kepercayaan pengguna terhadap model, diterapkan pendekatan interpretabilitas berbasis GraphSVX agar setiap prediksi dapat dijelaskan secara transparan.

### Tujuan Proyek
- Menerapkan GraphSAGE pada data keamanan siber untuk mendeteksi insider threat.
- Mengembangkan pendekatan interpretable AI melalui integrasi GraphSVX.
- Mengevaluasi performa model menggunakan metrik yang komprehensif.
- Menyediakan dokumentasi visual dan naratif sebagai bagian dari hasil riset.

### Fitur Utama
- Pemodelan graph heterogen dengan tiga tipe node utama: `user`, `pc`, dan `url`.
- Arsitektur multi-layer GraphSAGE untuk klasifikasi binary.
- Penanganan masalah class imbalance dengan teknik sampling dan pembobotan.
- Evaluasi performa model menggunakan AUC, Precision-Recall, dan analisis threshold optimal.
- Integrasi metode GraphSVX untuk meningkatkan interpretabilitas model.
- Visualisasi prediksi dan hasil interpretasi untuk mendukung analisis kualitatif.

### Struktur Proyek
```bash
data/                   # Dataset hasil preprocessing  
models/                 # Checkpoint hasil training model  
src/                    # Kode sumber (training, evaluasi, interpretasi, visualisasi)  
main.py                 # Pipeline utama yang mengintegrasikan seluruh proses  
README.md               # Dokumentasi proyek  
requirements.txt        # Daftar dependensi Python  
```

### Panduan Eksekusi
1. Buat virtual environment
```bash
python -m venv env
```
2. Aktifkan environment
```bash
# Windows
env\Scripts\activate

# Linux/macOS
source env/bin/activate
```
3. Instalasi dependensi
```bash
pip install -r requirements.txt
```
4. Jalankan pipeline utama (default:1000 users)
```bash
python main.py
```
**Opsional**: </br>
Jalankan pipeline utama dengan jumlah users tertentu
```bash
python main.py --users 1000    # Untuk 1000 users
python main.py --users 1200    # Untuk 1200 users
python main.py --users 1300    # Untuk 1300 users
python main.py --users 1400    # Untuk 1400 users
python main.py --users 1500    # Untuk 1500 users
python main.py --users 1600    # Untuk 1600 users
python main.py --users 1700    # Untuk 1700 users
python main.py --users 1800    # Untuk 1800 users
```
Jalankan bagian tertentu (default: 1000 users)
```bash
python -m core.train               # Melatih model
python -m core.evaluate            # Evaluasi model
python -m core.explain             # Penjelasan interpretasi model
python -m core.visual              # Visualisasi hasil
```
### Output Utama
1. `insider_threat_graphsage.pt` – Model terlatih untuk prediksi insider threat.
2. `evaluation_results.pkl` – File berisi metrik evaluasi (AUC, precision, recall, dsb).
3. `graphsvx_explanations.pkl` – Hasil interpretasi prediksi model untuk pengguna berisiko.

### Pustaka yang Digunakan
- PyTorch & PyTorch Geometric – Untuk implementasi GraphSAGE
- scikit-learn – Untuk evaluasi dan metrik model
- matplotlib & seaborn – Untuk visualisasi eksploratif

### Referensi Ilmiah
- Hamilton, W. dkk. (2017). [*Inductive Representation Learning on Large Graphs* (GraphSAGE)](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).  
- Duval, A. dkk. (2021). [*GraphSVX: Shapley Value Explanations for Graph Neural Networks*](https://arxiv.org/pdf/2104.10482).  
- CERT Carnegie Mellon. [*Insider Threat Dataset*](https://kilthub.cmu.edu/ndownloader/files/24844280)

### Penulis
Tugas Akhir oleh [Aaz Zazri Nugraha](https://github.com/azzazry)  
Universitas Jenderal Achmad Yani Yogyakarta (2025)  
_Email: azzazry120@email.com_
