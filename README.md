# GNN Project: GraphSAGE & GraphSVX
Proyek ini merupakan bagian dari tugas akhir sarjana pada bidang Teknologi Informasi. Implementasi ini berfokus pada pemanfaatan _Graph Neural Network_ (GNN) untuk mendeteksi anomali berbasis graf heterogen, serta memberikan penjelasan terhadap prediksi model menggunakan pendekatan GraphSVX. Inspirasi awal berasal dari repositori [github.com/AlexDuvalinho/GraphSVX)](https://github.com/AlexDuvalinho/GraphSVX), dengan sejumlah penyesuaian untuk konteks keamanan siber.

### Abstrak
Anomali dalam sistem komputer, seperti perilaku insider threat, merupakan ancaman yang sulit dideteksi dengan pendekatan konvensional. _Graph Neural Network_ (GNN), khususnya GraphSAGE, menawarkan pendekatan yang efisien untuk memahami struktur relasi dalam data berbasis graf. Proyek ini mengembangkan sistem klasifikasi binary berbasis graph heterogen (user, PC, URL), disertai metode interpretabilitas menggunakan GraphSVX, agar hasil prediksi model dapat dipahami secara transparan oleh pengguna akhir.

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
4. Jalankan pipeline utama
```bash
python main.py
```
5. Jalankan bagian tertentu (opsional)
```bash
python src.train               # Melatih model
python src.evaluate            # Evaluasi model
python src.explain             # Penjelasan interpretasi model
python src.visual              # Visualisasi hasil
```
### Output Utama
1. `insider_threat_graphsage.pt` – File model hasil training
2. `evaluation_results.pkl` – Ringkasan metrik evaluasi
3. `graphsvx_explanations.pkl` – Hasil interpretasi dengan GraphSVX

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