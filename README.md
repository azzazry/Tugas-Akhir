
# Insider Threat Detection dengan GraphSAGE & GraphSVX

## Deskripsi  
Proyek ini bertujuan mendeteksi ancaman dari dalam (insider threat) di lingkungan organisasi menggunakan Graph Neural Networks, khususnya GraphSAGE, dan menjelaskan prediksi model dengan GraphSVX.

## Fitur Utama  
- Model heterogeneous graph dengan node user, PC, file, dan URL  
- Multi-layer GraphSAGE untuk klasifikasi binary  
- Penanganan class imbalance (imbalan kelas)  
- Evaluasi lengkap dengan AUC, Precision-Recall, dan threshold optimal  
- Interpretabilitas model dengan GraphSVX  
- Visualisasi hasil prediksi dan penjelasan model

## Struktur Direktori  
```
data/                   # Dataset graph    
models/                 # Model checkpoint  
src/                    # Kode sumber (model, training, evaluasi, visualisasi)  
main.py                 # Pipeline eksekusi seluruh proses  
README.md               # Dokumentasi ini  
requirements.txt        # Dependensi Python  
```

## Cara Menjalankan  
1. Install dependencies:  
```bash
pip install -r requirements.txt
```

2. Jalankan pipeline:  
```bash
python main.py
```

## Output Penting  
- `insider_threat_graphsage.pt` — Model hasil training  
- `evaluation_results.pkl` — Hasil evaluasi model  
- `graphsvx_explanations.pkl` — Penjelasan model dengan GraphSVX  

## Tools & Library  
- PyTorch & PyTorch Geometric  
- GraphSVX  
- scikit-learn  
- matplotlib & seaborn  

## Referensi  
- [GraphSAGE Paper](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)  
- [GraphSVX Paper](https://arxiv.org/pdf/2104.10482)  
- [CERT Dataset](https://kilthub.cmu.edu/ndownloader/files/24844280)

## Author  
Tugas Akhir oleh [Azzazry](https://github.com/azzazry)