# Deteksi Insider Threat Menggunakan Graph Neural Network (GraphSAGE) dan Interpretasi GraphSVX
Proyek ini merupakan bagian dari tugas akhir sarjana di bidang Teknologi Informasi. Fokus utamanya adalah deteksi anomali perilaku karyawan di lingkungan korporat (_insider threat_), yang kerap sulit terdeteksi oleh metode konvensional. Implementasi ini memadukan metode _Graph Neural Network_ (GNN), khususnya arsitektur GraphSAGE, dengan pendekatan interpretasi berbasis GraphSVX sederhana yang telah dimodifikasi agar sesuai dengan konteks keamanan siber dan kebutuhan riset ini.

### Abstrak
Anomali seperti _insider threat_ sering kali tersamar dalam aktivitas normal sistem, menjadikannya sulit dikenali. GNN, khususnya GraphSAGE, mampu merepresentasikan hubungan kompleks dalam data berbasis graf. Dalam proyek ini, dikembangkan sistem klasifikasi biner pada graf heterogen yang terdiri dari node bertipe `user`, `pc`, dan `url`. Agar hasil prediksi tidak hanya akurat tapi juga transparan, digunakan metode interpretasi berbasis adaptasi GraphSVX sederhana untuk menjelaskan kontribusi fitur terhadap keputusan model.

### Tujuan Proyek
* Menerapkan GraphSAGE pada data keamanan siber untuk mendeteksi insider threat.
* Mengintegrasikan pendekatan interpretable AI melalui adaptasi GraphSVX.
* Mengevaluasi performa model menggunakan metrik evaluasi komprehensif.
* Menyediakan dokumentasi visual dan naratif untuk keperluan riset.

### Fitur Utama
* Arsitektur GraphSAGE multilayer untuk klasifikasi biner.
* Evaluasi performa menggunakan AUC, precision-recall, dan threshold analysis.
* Interpretasi model berbasis GraphSVX (versi sederhana).
* Visualisasi prediksi dan interpretasi hasil untuk insight kualitatif.

### Struktur Proyek
```bash
├── core/                       # Script utama untuk training, evaluasi, interpretasi, visualisasi
├── data/                       # Dataset hasil preprocessing
├── src/
│   ├── models/                 # Model GraphSAGE & GraphSVX
│   ├── plots/                  # Grafik visualisasi hasil
│   └── utils/                  # Fungsi bantu
├── main.py                     # Pipeline utama
```

### Panduan Eksekusi
Ikuti langkah-langkah berikut untuk menjalankan pipeline deteksi insider threat:
#### 1. Membuat dan Mengaktifkan Virtual Environment
Langkah ini bertujuan untuk memastikan seluruh dependensi proyek terisolasi dari sistem global.
```bash
# Buat virtual environment
python -m venv env
```
Aktifkan environment sesuai sistem operasi:
```bash
# Windows
env\Scripts\activate

# Linux/macOS
source env/bin/activate
```
#### 2. Install Dependensi
Instal seluruh dependensi proyek dari file `requirements.txt`:
```bash
pip install -r requirements.txt
```
#### 3. Menjalankan Pipeline Utama
Pipeline utama dijalankan melalui file `main.py`, yang akan melaksanakan seluruh tahapan mulai dari pelatihan model hingga visualisasi hasil.
```bash
python main.py
```
Secara default, pipeline akan menggunakan 1000 user pertama dari dataset.
### Opsi Tambahan (Opsional)
#### Menentukan jumlah pengguna (`--users`)
```bash
python main.py --users 1200
```
Menjalankan pipeline dengan 1200 pengguna pertama dari dataset.
#### Menentukan jumlah top user berisiko tertinggi (`--top_n`)
```bash
python main.py --top_n 2
```
Menampilkan interpretasi untuk 2 pengguna dengan skor risiko tertinggi. Nilai dapat berupa angka atau "all".
#### Kombinasi argumen `--users` dan `--top_n`
```bash
python main.py --users 1400 --top_n 3
```
Menjalankan pipeline untuk 1400 user dan menampilkan 3 interpretasi pengguna paling berisiko.
### Menjalankan Bagian Pipeline Secara Terpisah
Bagian-bagian pipeline dapat dieksekusi secara modular sesuai kebutuhan:
```bash
python -m core.train           # Hanya training model
python -m core.evaluate        # Evaluasi hasil prediksi
python -m core.explain         # Interpretasi model dengan GraphSVX sederhana
python -m core.visual          # Visualisasi metrik dan interpretasi
```
### Output Utama
1. `models/insider_threat_graphsage.pt` – Model terlatih.
2. `result/evaluation_results.pkl` – Metrik evaluasi model.
3. `result/graphsvx_explanations.pkl` – Interpretasi prediksi berdasarkan top\_n user berisiko.
4. `result/visualizations/` – Grafik visualisasi hasil evaluasi dan interpretasi.
### Dependensi Utama
* **PyTorch & PyTorch Geometric** – Untuk implementasi GraphSAGE.
* **scikit-learn** – Untuk evaluasi model dan perhitungan metrik.
* **matplotlib & seaborn** – Untuk visualisasi data dan hasil model.

### Referensi Ilmiah
* Hamilton, W. dkk. (2017). *[Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)*.
* Duval, A. dkk. (2021). *[GraphSVX: Shapley Value Explanations for Graph Neural Networks](https://arxiv.org/pdf/2104.10482)*.
* CERT Carnegie Mellon. *[Insider Threat Dataset](https://kilthub.cmu.edu/ndownloader/files/24844280)*

### Penulis

Tugas Akhir oleh [Aaz Zazri Nugraha](https://github.com/azzazry)
Universitas Jenderal Achmad Yani Yogyakarta (2025)
*Email: [azzazry120@email.com](mailto:azzazry120@email.com)*