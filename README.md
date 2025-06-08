# GNN Project: GraphSAGE & GraphSVX
Proyek ini merupakan bagian dari skripsi dan terinspirasi oleh repositori [GraphSVX](https://github.com/AlexDuvalinho/GraphSVX), dengan beberapa penyesuaian dan pengembangan tambahan.

### Abstak
Pemanfaatan Graph Neural Network (GNN) sebagai model untuk mendeteksi anomali terbukti cukup powerful, terutama untuk data yang bersifat relasional seperti insider threat. Proyek ini mengimplementasikan heterogeneous GraphSAGE untuk klasifikasi binary serta memberikan interpretabilitas model menggunakan GraphSVX.

### Fitur Utama  
- Model heterogeneous graph dengan node user, PC, dan URL  
- Multi-layer GraphSAGE untuk klasifikasi binary  
- Penanganan class imbalance (imbalan kelas) menggunakan random oversampling
- Evaluasi lengkap dengan AUC, Precision-Recall, dan threshold optimal  
- Interpretabilitas model dengan GraphSVX  
- Visualisasi hasil prediksi dan penjelasan model

### Struktur Direktori  
```
data/                   # Dataset graph    
models/                 # Model checkpoint  
src/                    # Kode sumber (model, training, evaluasi, visualisasi)  
main.py                 # Pipeline eksekusi seluruh proses  
README.md               # Dokumentasi ini  
requirements.txt        # Dependensi Python  
```

### Cara Menjalankan  
1. Buat virtual environment:
```bash
python -m venv env
```
2. Jalankan virtual environment:
```bash
# Windows
/env/Scripts/activate

# Linux/MacOS
source env/bin/activate
```

2. Install dependencies:  
```bash
pip install -r requirements.txt
```

3. Jalankan pipeline:  
```bash
python main.py
```
4. Jika ingin menjalankan file tertentu
```bash
python src.train               # Training model
python src.evaluate            # Evaluasi model
python src.explain_graphsvx    # Interpretasi dengan GraphSVX
python src.visualization       # Visualisasi hasil
```

### Output Penting  
- `insider_threat_graphsage.pt` — Model hasil training  
- `evaluation_results.pkl` — Hasil evaluasi model  
- `graphsvx_explanations.pkl` — Penjelasan model dengan GraphSVX  

### Tools & Library  
- PyTorch & PyTorch Geometric  
- scikit-learn  
- matplotlib & seaborn  

### Referensi
- [GraphSAGE Paper](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)  
- [GraphSVX Paper](https://arxiv.org/pdf/2104.10482)  
- [CERT Dataset](https://kilthub.cmu.edu/ndownloader/files/24844280)

### Author  
Tugas Akhir oleh [Aaz Zazri Nugraha](https://github.com/azzazry)<br>
_“Keep it real, keep it explainable.”_