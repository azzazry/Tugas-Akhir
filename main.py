import os
import traceback
import argparse

from src.train import train_insider_threat_model
from src.evaluate import evaluate_insider_threat_model
from src.explain import explain_insider_predictions
from src.visual import create_research_visualizations

def create_directories(users):
    os.makedirs(f'result/{users}_users/visualizations', exist_ok=True)
    os.makedirs(f'result/{users}_users/logs', exist_ok=True)
    os.makedirs(f'result/{users}_users/data', exist_ok=True)

def run_pipeline(users):
    try:
        print(f"\n[1] Mempersiapkan direktori untuk {users} users...")
        create_directories(users)

        print("\n[2] Training model Insider Threat GraphSAGE...")
        train_insider_threat_model(users)

        print("\n[3] Mengevaluasi performa model...")
        evaluate_insider_threat_model(users)

        print("\n[4] Menjalankan interpretasi GraphSVX pada prediksi berisiko...")
        explain_insider_predictions(users)

        print("\n[5] Membuat visualisasi hasil evaluasi dan interpretasi...")
        create_research_visualizations(users)

        print("\nPipeline selesai dijalankan tanpa error.")
    except Exception as e:
        print("\nTerjadi error saat menjalankan pipeline:")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--users', type=str, default='1000', help="Jumlah user (1000 atau 1500)")
    args = parser.parse_args()
    run_pipeline(args.users)