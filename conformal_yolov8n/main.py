from src.config import RESULTS_DIR, TABLES_DIR
from src.data_utils import ensure_dir

def main():
    ensure_dir(RESULTS_DIR)
    ensure_dir(TABLES_DIR)
    print("Conformal YOLOv8n project setup is ready.")

if __name__ == "__main__":
    main()
