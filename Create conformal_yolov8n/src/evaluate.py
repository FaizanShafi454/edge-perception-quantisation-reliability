import time
import pandas as pd

def benchmark_model(model, source, conf=0.25, imgsz=640):
    start = time.perf_counter()
    results = model.predict(source=source, conf=conf, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start
    return results, elapsed

def save_results(rows, output_path):
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
