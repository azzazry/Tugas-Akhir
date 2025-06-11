import os
import traceback

from src.train import train_insider_threat_model
from src.evaluate import evaluate_insider_threat_model
from src.explain import explain_insider_predictions
from src.visual import create_research_visualizations

def create_directories():
    os.makedirs('result/visualizations', exist_ok=True)
    os.makedirs('result/logs', exist_ok=True)
    os.makedirs('result/data', exist_ok=True)

def run_pipeline():
    try:
        print("\n[1] Mempersiapkan direktori...")
        create_directories()

        print("\n[2] Training model Insider Threat GraphSAGE...")
        train_insider_threat_model()

        print("\n[3] Mengevaluasi performa model...")
        evaluate_insider_threat_model()

        print("\n[4] Menjalankan interpretasi GraphSVX pada prediksi berisiko...")
        explain_insider_predictions()

        print("\n[5] Membuat visualisasi hasil evaluasi dan interpretasi...")
        create_research_visualizations()

        print("\nPipeline selesai dijalankan tanpa error.")
    except Exception as e:
        print("\nTerjadi error saat menjalankan pipeline:")
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()