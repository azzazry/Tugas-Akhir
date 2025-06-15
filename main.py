import os
import traceback

from core.train import train_insider_threat_model
from core.evaluate import evaluate_insider_threat_model
from core.explain import explain_insider_predictions
from core.visual import create_research_visualizations

try:
    from src.utils.argparse import get_arguments
except ImportError:
    get_arguments = None

def create_directories(users):
    os.makedirs(f'result/{users}_users/visualizations', exist_ok=True)
    os.makedirs(f'result/{users}_users/logs', exist_ok=True)
    os.makedirs(f'result/{users}_users/data', exist_ok=True)

def run_pipeline(users, top_n):
    try:
        print(f"\n[1] Mempersiapkan direktori untuk {users} users...")
        create_directories(users)

        print("\n[2] Training model Insider Threat GraphSAGE...")
        train_insider_threat_model(users)

        print("\n[3] Mengevaluasi performa model...")
        evaluate_insider_threat_model(users)

        print("\n[4] Menjalankan interpretasi GraphSVX pada prediksi berisiko...")
        explain_insider_predictions(users, top_n=top_n)

        print("\n[5] Membuat visualisasi hasil evaluasi dan interpretasi...")
        create_research_visualizations(users, top_n=top_n)

        print("\nPipeline selesai dijalankan tanpa error.")
    except Exception:
        print("\nTerjadi error saat menjalankan pipeline:")
        traceback.print_exc()

if __name__ == "__main__":
    if get_arguments:
        args = get_arguments()
        run_pipeline(users=args.users, top_n=args.top_n)
    else:
        run_pipeline(users='1000', top_n='5')