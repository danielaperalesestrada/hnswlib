import chromadb
import numpy as np
import time
import logging
import warnings
import hnswlib
import os
import urllib.request
import h5py

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

N = 1000 
N_D = 20000
NQ = 100
K = 10
DIM = 128
REPETICIONES = 5
REPETICIONES_B = 20
CHROMA_MAX_BATCH = 5000

SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
SIFT_PATH = "sift-128-euclidean.hdf5"


def generate_uniform(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim)).astype(np.float32)


def generate_clustered(n, dim, n_clusters=20, sigma=0.05, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.random((n_clusters, dim)).astype(np.float32)
    data = []
    for _ in range(n):
        c = rng.integers(0, n_clusters)
        point = centers[c] + rng.normal(0, sigma, dim).astype(np.float32)
        point = np.clip(point, 0, 1)
        data.append(point)
    return np.array(data)


def download_sift_if_needed(path=SIFT_PATH, url=SIFT_URL):
    if os.path.exists(path):
        return path
    print(f"Descargando SIFT desde {url} (esto puede tardar unos minutos)...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(path, "wb") as out_file:
        total = response.length or 0
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"  Descargado: {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.1f}%)", end="\r", flush=True)
    print()
    return path


def load_sift_subset(n_base, n_queries, path=SIFT_PATH, seed=42):
    download_sift_if_needed(path)
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        total_base = f["train"].shape[0]
        total_queries = f["test"].shape[0]

        base_idx = rng.choice(total_base, size=n_base, replace=False)
        base_idx.sort()
        base = f["train"][base_idx].astype(np.float32)

        query_idx = rng.choice(total_queries, size=n_queries, replace=False)
        query_idx.sort()
        queries = f["test"][query_idx].astype(np.float32)

    return base, queries


def compute_ground_truth(base, queries, k):
    gt = []
    for q in queries:
        dists = np.sum((base - q) ** 2, axis=1)
        top_k = set(np.argsort(dists)[:k].tolist())
        gt.append(top_k)
    return gt


def run_benchmark_original(nombre, base, queries, gt, reps=REPETICIONES):
    recalls = []
    tiempos_insercion = []
    tiempos_busqueda = []

    for rep in range(reps):
        index = hnswlib.Index(space='l2', dim=base.shape[1])
        index.init_index(max_elements=len(base), ef_construction=100, M=16)
        index.set_ef(10)

        ids = np.arange(len(base))

        t0 = time.time()
        index.add_items(base, ids)
        tiempos_insercion.append(time.time() - t0)

        t0 = time.time()
        all_labels, _ = index.knn_query(queries, k=K, num_threads=1)
        all_labels = [row.tolist() for row in all_labels]
        tiempos_busqueda.append((time.time() - t0) / len(queries) * 1000)

        correct = 0
        for i, result_ids in enumerate(all_labels):
            found = set(int(r) for r in result_ids)
            correct += len(found & gt[i])
        recalls.append(correct / (len(queries) * K))

        print(f"  Rep {rep+1}/{reps} — Recall: {recalls[-1]:.4f}", flush=True)

    print(f"\n{'='*50}")
    print(f"Escenario ORIGINAL: {nombre}")
    print(f"{'='*50}")
    print(f"  Inserción promedio : {np.mean(tiempos_insercion):.3f}s")
    print(f"  Búsqueda promedio  : {np.mean(tiempos_busqueda):.3f}ms")
    print(f"  Recall@{K} promedio : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  Recall@{K} mínimo  : {np.min(recalls):.4f}")
    print(f"  Recall@{K} máximo  : {np.max(recalls):.4f}")

    return np.mean(recalls), np.std(recalls), np.mean(tiempos_insercion), np.mean(tiempos_busqueda)


def run_benchmark(nombre, base, queries, gt, reps=REPETICIONES):
    recalls = []
    tiempos_insercion = []
    tiempos_busqueda = []

    for rep in range(reps):
        client = chromadb.Client()
        col_name = f"bench{rep}{int(time.time()) % 100000}"
        collection = client.create_collection(name=col_name)

        ids = [str(i) for i in range(len(base))]
        embeddings = base.tolist()
        documents = [f"doc_{i}" for i in range(len(base))]

        t0 = time.time()
        for start in range(0, len(base), CHROMA_MAX_BATCH):
            end = start + CHROMA_MAX_BATCH
            collection.add(
                embeddings=embeddings[start:end],
                ids=ids[start:end],
                documents=documents[start:end],
            )
        tiempos_insercion.append(time.time() - t0)

        t0 = time.time()
        results = collection.query(
            query_embeddings=queries.tolist(),
            n_results=K
        )
        tiempos_busqueda.append((time.time() - t0) / len(queries) * 1000)

        correct = 0
        for i, result_ids in enumerate(results['ids']):
            found = set(int(r) for r in result_ids)
            correct += len(found & gt[i])
        recalls.append(correct / (len(queries) * K))

        print(f"  Rep {rep+1}/{reps} — Recall: {recalls[-1]:.4f}", flush=True)

    print(f"\n{'='*50}")
    print(f"Escenario: {nombre}")
    print(f"{'='*50}")
    print(f"  Inserción promedio : {np.mean(tiempos_insercion):.3f}s")
    print(f"  Búsqueda promedio : {np.mean(tiempos_busqueda):.3f}ms")
    print(f"  Recall@{K} promedio : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  Recall@{K} mínimo : {np.min(recalls):.4f}")
    print(f"  Recall@{K} máximo : {np.max(recalls):.4f}")

    return np.mean(recalls), np.std(recalls), np.mean(tiempos_insercion), np.mean(tiempos_busqueda)


print("=" * 50)
print("  Benchmark Chroma + HierarchicalNSWAdaptive")
print("=" * 50)
print(f"  N={N}, NQ={NQ}, K={K}, DIM={DIM}, Reps={REPETICIONES}")

print("\nGenerando escenario A (uniforme dim=128)...")
base_a = generate_uniform(N, DIM, seed=42)
queries_a = generate_uniform(NQ, DIM, seed=99)
gt_a = compute_ground_truth(base_a, queries_a, K)
recall_a, std_a, t_ins_a, t_bus_a = run_benchmark("A Uniforme 128", base_a, queries_a, gt_a)

print("\nGenerando escenario B (clusters dim=128)...")
base_b = generate_clustered(N, DIM, seed=42)
queries_b = generate_clustered(NQ, DIM, seed=99)
gt_b = compute_ground_truth(base_b, queries_b, K)
recall_b, std_b, t_ins_b, t_bus_b = run_benchmark("B Clusters 128", base_b, queries_b, gt_b, reps=REPETICIONES_B)

print("\nGenerando escenario C (uniforme dim=32)...")
base_c = generate_uniform(N, 32, seed=42)
queries_c = generate_uniform(NQ, 32, seed=99)
gt_c = compute_ground_truth(base_c, queries_c, K)
recall_c, std_c, t_ins_c, t_bus_c = run_benchmark("C Uniforme 32", base_c, queries_c, gt_c)

print("\nGenerando escenario D (SIFT real, subset dim=128)...")
base_d, queries_d = load_sift_subset(n_base=N_D, n_queries=NQ, seed=42)
gt_d = compute_ground_truth(base_d, queries_d, K)
recall_d, std_d, t_ins_d, t_bus_d = run_benchmark("D SIFT real 128", base_d, queries_d, gt_d)

print(f"\n{'='*55}")
print("  RESUMEN FINAL")
print(f"{'='*55}")
print(f"{'Escenario':<22} {'Recall@10':<12} {'Inserción':<12} {'Búsqueda'}")
print(f"{'-'*55}")
print(f"{'A: Uniforme dim=128':<22} {recall_a:<12.4f} {t_ins_a:<10.3f}s {t_bus_a:.3f}ms")
print(f"{'B: Clusters dim=128':<22} {recall_b:<12.4f} {t_ins_b:<10.3f}s {t_bus_b:.3f}ms")
print(f"{'C: Uniforme dim=32':<22} {recall_c:<12.4f} {t_ins_c:<10.3f}s {t_bus_c:.3f}ms")
print(f"{'D: SIFT real dim=128':<22} {recall_d:<12.4f} {t_ins_d:<10.3f}s {t_bus_d:.3f}ms")

print("\n\n" + "=" * 55)
print("  BENCHMARK ORIGINAL (hnswlib directo, sin Chroma)")
print("=" * 55)

print("\nGenerando escenario A (uniforme dim=128)...")
ro_a, sro_a, to_ins_a, to_bus_a = run_benchmark_original("A Uniforme 128", base_a, queries_a, gt_a)

print("\nGenerando escenario B (clusters dim=128)...")
ro_b, sro_b, to_ins_b, to_bus_b = run_benchmark_original("B Clusters 128", base_b, queries_b, gt_b, reps=REPETICIONES_B)

print("\nGenerando escenario C (uniforme dim=32)...")
ro_c, sro_c, to_ins_c, to_bus_c = run_benchmark_original("C Uniforme 32", base_c, queries_c, gt_c)

print("\nGenerando escenario D (SIFT real, subset dim=128)...")
ro_d, sro_d, to_ins_d, to_bus_d = run_benchmark_original("D SIFT real 128", base_d, queries_d, gt_d)

print(f"\n{'='*65}")
print("  COMPARACIÓN FINAL: Adaptativo (Chroma) vs Original (directo)")
print(f"{'='*65}")
print(f"  (B usa {REPETICIONES_B} repeticiones; A, C, D usan {REPETICIONES})")
print(f"{'-'*65}")
print(f"{'Escenario':<22} {'Recall Adapt':>13} {'Recall Orig':>12} {'Diferencia':>11}")
print(f"{'-'*65}")
print(f"{'A: Uniforme dim=128':<22} {recall_a:>13.4f} {ro_a:>12.4f} {recall_a-ro_a:>+11.4f}")
print(f"{'B: Clusters dim=128':<22} {recall_b:>13.4f} {ro_b:>12.4f} {recall_b-ro_b:>+11.4f}")
print(f"    (std Adapt ± {std_b:.4f} | std Orig ± {sro_b:.4f})")
print(f"{'C: Uniforme dim=32':<22} {recall_c:>13.4f} {ro_c:>12.4f} {recall_c-ro_c:>+11.4f}")
print(f"{'D: SIFT real dim=128':<22} {recall_d:>13.4f} {ro_d:>12.4f} {recall_d-ro_d:>+11.4f}")

se_adapt_b = std_b / np.sqrt(REPETICIONES_B)
se_orig_b = sro_b / np.sqrt(REPETICIONES_B)
se_diff_b = np.sqrt(se_adapt_b**2 + se_orig_b**2)
diff_b = recall_b - ro_b
t_stat_b = diff_b / se_diff_b if se_diff_b > 0 else float('nan')

print(f"\n  Chequeo estadístico en B (Welch t-test aproximado):")
print(f"    diferencia de promedios = {diff_b:+.4f}")
print(f"    error estándar de la diferencia = ±{se_diff_b:.4f}")
print(f"    t ≈ {t_stat_b:.2f}  (|t| > ~2 sugiere diferencia real, no ruido)")
if abs(t_stat_b) > 2:
    print("  -> La diferencia en B es estadísticamente consistente, no parece ser solo ruido.")
else:
    print("  -> La diferencia en B todavía no se distingue con confianza del ruido.")