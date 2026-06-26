// bench_comparison.cpp — v5b
//
// Benchmark: HierarchicalNSW (original) vs HierarchicalNSWAdaptive v5
//
// Ejecuta los mismos 9 experimentos sobre tres escenarios distintos:
//
//   Escenario A — Uniforme dim=128
//     Todos los puntos uniformes en [0,1]^128.
//     Contexto local homogéneo: AdaptiveState siempre ~0.5.
//     M1-M3 apenas se activan. Línea base para comparar overhead.
//
//   Escenario B — Clusters dim=128
//     20 clusters gaussianos, sigma=0.05, en [0,1]^128.
//     Densidad intra-cluster ~67x mayor que uniforme.
//     AdaptiveState detecta density_score~1 dentro de clusters,
//     ~0 en zonas inter-cluster: M1-M3 deberían activarse.
//
//   Escenario C — Uniforme dim=32
//     Dimensión baja: la mejora angular (M3) puede activarse (dim<=64).
//     Menor "maldición de la dimensionalidad": recall base más alto.
//
// Compilar:
//   g++ -O3 -std=c++17 -I. bench_comparison.cpp -o bench_comparison
//
// Ejecutar:
//   ./bench_comparison              (todos los escenarios, defaults)
//   ./bench_comparison [n] [nq]     (ajustar tamaño)

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <functional>

#include "../hnswlib/hnswlib.h"
#include "../hnswlib/hnswalg_adaptive.h"

using Clock  = std::chrono::high_resolution_clock;
using Matrix = std::vector<std::vector<float>>;

// =============================================================================
// Generadores de datos
// =============================================================================

// Vectores uniformes en [0,1]^dim
Matrix generateUniform(size_t n, size_t dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Matrix data(n, std::vector<float>(dim));
    for (auto& v : data) for (auto& x : v) x = dist(rng);
    return data;
}

// Mezcla gaussiana: n_clusters centros, puntos con ruido sigma
// Los centros se generan uniformes en [0,1]^dim.
// Cada punto se asigna aleatoriamente a un cluster.
Matrix generateClustered(size_t n, size_t dim,
                          size_t n_clusters = 20, float sigma = 0.05f,
                          unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);
    std::normal_distribution<float>       ndist(0.0f, sigma);
    std::uniform_int_distribution<size_t> cdist(0, n_clusters - 1);

    // Generar centros
    Matrix centers(n_clusters, std::vector<float>(dim));
    for (auto& c : centers) for (auto& x : c) x = udist(rng);

    // Generar puntos alrededor de centros
    Matrix data(n, std::vector<float>(dim));
    for (auto& v : data) {
        size_t ci = cdist(rng);
        for (size_t d = 0; d < dim; d++)
            v[d] = std::max(0.0f, std::min(1.0f, centers[ci][d] + ndist(rng)));
    }
    return data;
}

// =============================================================================
// Ground truth por fuerza bruta (L2²)
// =============================================================================
std::vector<std::unordered_set<size_t>>
computeGroundTruth(const Matrix& base, const Matrix& queries, size_t k) {
    size_t nq = queries.size(), nb = base.size(), dim = base[0].size();
    std::vector<std::unordered_set<size_t>> gt(nq);
    for (size_t qi = 0; qi < nq; qi++) {
        std::vector<std::pair<float,size_t>> dists(nb);
        for (size_t bi = 0; bi < nb; bi++) {
            float d = 0.0f;
            for (size_t di = 0; di < dim; di++) {
                float t = queries[qi][di] - base[bi][di]; d += t * t;
            }
            dists[bi] = {d, bi};
        }
        std::partial_sort(dists.begin(), dists.begin() + (int)k, dists.end());
        for (size_t i = 0; i < k; i++) gt[qi].insert(dists[i].second);
    }
    return gt;
}

// =============================================================================
// Benchmark genérico
// =============================================================================
template <typename IndexT>
void runBench(
    const std::string& label,
    IndexT& idx,
    const Matrix& base,
    const Matrix& queries,
    const std::vector<std::unordered_set<size_t>>& gt,
    size_t k)
{
    size_t nq = queries.size();

    // Flush de caché de CPU entre experimentos para reducir varianza en build.
    // Escribe y lee 32 MB para desalojar el índice anterior de L3.
    {
        volatile char sink = 0;
        std::vector<char> flush_buf(32 * 1024 * 1024, 1);
        for (auto b : flush_buf) sink += b;
        (void)sink;
    }

    auto t0 = Clock::now();
    for (size_t i = 0; i < base.size(); i++)
        idx.addPoint(base[i].data(), i);
    double build_ms = std::chrono::duration<double,std::milli>(
        Clock::now() - t0).count();

    idx.metric_distance_computations = 0;
    size_t correct = 0;
    std::vector<double> lat(nq);

    for (size_t qi = 0; qi < nq; qi++) {
        auto ts = Clock::now();
        auto res = idx.searchKnn(queries[qi].data(), k);
        lat[qi] = std::chrono::duration<double,std::micro>(
            Clock::now() - ts).count();
        while (!res.empty()) {
            if (gt[qi].count(res.top().second)) correct++;
            res.pop();
        }
    }

    double mean_us = std::accumulate(lat.begin(), lat.end(), 0.0)
                     / static_cast<double>(nq);
    std::sort(lat.begin(), lat.end());
    double p99_us = lat[static_cast<size_t>(nq * 0.99)];
    double recall = static_cast<double>(correct) / static_cast<double>(nq * k);
    double nodes  = static_cast<double>(idx.metric_distance_computations)
                  / static_cast<double>(nq);

    std::cout << std::left  << std::setw(28) << label
              << std::right << std::fixed
              << std::setw(11) << std::setprecision(1) << build_ms
              << std::setw(12) << std::setprecision(1) << mean_us
              << std::setw(10) << std::setprecision(1) << p99_us
              << std::setw(10) << std::setprecision(4) << recall
              << std::setw(14) << std::setprecision(1) << nodes
              << "\n";
}

// =============================================================================
// Helpers de configuración
// =============================================================================
using Cfg = hnswlib::HierarchicalNSWAdaptive<float>::AdaptiveConfig;

// cfgNone: apaga todas las mejoras adaptativas.
// Debe cubrir TODOS los campos nuevos de v6 para que needsConstructionProbe()
// retorne false y M4-solo no pague el costo del probe en construcción.
Cfg cfgNone() {
    Cfg c;
    // M1 off
    c.ml_scale_min = c.ml_scale_max = 1.0f;
    // M2 off — incluye campos nuevos de v6
    c.ef_c_density_gain  = 0.0f;
    c.ef_c_variance_gain = 0.0f;
    c.ef_c_skewness_gain = 0.0f;   // v6: campo nuevo
    c.ef_c_frontier_boost = 0.0f;  // v6: campo nuevo
    // M3 off
    c.angular_max_dim = 0;
    // M4 off — ef_s_scale_min=max=1.0 → escala neutra
    c.ef_s_scale_min      = 1.0f;
    c.ef_s_scale_max      = 1.0f;
    c.ef_s_skewness_boost = 0.0f;  // v6: campo nuevo
    // Pesos de difficulty (irrelevantes cuando todo off, pero explícitos)
    c.w_density  = 0.4f;
    c.w_variance = 0.4f;
    c.w_skewness = 0.2f;           // v6: campo nuevo
    return c;
}

Cfg cfgFull()  { return Cfg{}; }  // defaults v6: todo activo

// M1: revSize adaptativo — disperso → jerarquía más profunda
Cfg cfgM1()    { auto c=cfgNone(); c.ml_scale_max=1.2f; return c; }

// M2: ef_construction adaptativo — incluye skewness y frontier boost (v6)
Cfg cfgM2()    { auto c=cfgNone();
                 c.ef_c_density_gain  = 0.5f;
                 c.ef_c_variance_gain = 0.3f;
                 c.ef_c_skewness_gain = 0.3f;   // v6
                 c.ef_c_frontier_boost = 0.5f;  // v6
                 return c; }

// M3: diversificación angular — umbral dinámico por density_score
Cfg cfgM3()    { auto c=cfgNone();
                 c.angular_thresh_base=0.98f; c.angular_thresh_delta=0.10f;
                 c.angular_max_dim=64;
                 return c; }

// M4: ef_search adaptativo — incluye skewness boost (v6), piso=1.0 (N4)
Cfg cfgM4()    { auto c=cfgNone();
                 c.ef_s_scale_min=1.0f; c.ef_s_scale_max=2.5f;
                 c.ef_s_skewness_boost=0.5f;   // v6
                 return c; }

Cfg cfgM1M4()  { auto c=cfgM1();
                 c.ef_s_scale_min=1.0f; c.ef_s_scale_max=2.5f;
                 c.ef_s_skewness_boost=0.5f;
                 return c; }

Cfg cfgM2M4()  { auto c=cfgM2();
                 c.ef_s_scale_min=1.0f; c.ef_s_scale_max=2.5f;
                 c.ef_s_skewness_boost=0.5f;
                 return c; }

Cfg cfgM124()  { auto c=cfgM1M4();
                 c.ef_c_density_gain=0.5f; c.ef_c_variance_gain=0.3f;
                 c.ef_c_skewness_gain=0.3f; c.ef_c_frontier_boost=0.5f;
                 return c; }

// =============================================================================
// Ejecutar todos los experimentos para un escenario dado
// =============================================================================
void runScenario(
    const std::string& title,
    const Matrix& base, const Matrix& queries,
    const std::vector<std::unordered_set<size_t>>& gt,
    size_t n, size_t dim, size_t k,
    size_t M, size_t ef_c, size_t ef_s)
{
    const std::string sep87(85, '-');
    const std::string sep87d(85, '.');

    std::cout << "\n" << std::string(85, '=') << "\n"
              << "  " << title << "\n"
              << std::string(85, '=') << "\n";

    std::cout << std::left  << std::setw(28) << "Variante"
              << std::right
              << std::setw(11) << "Build(ms)"
              << std::setw(12) << "Search(us)"
              << std::setw(10) << "p99(us)"
              << std::setw(10) << "Recall@k"
              << std::setw(14) << "Nodes/query"
              << "\n" << sep87 << "\n";

    hnswlib::L2Space space(dim);

    // ── Linea base: HNSW original ────────────────────────────────────────────
    {
        hnswlib::HierarchicalNSW<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s);
        runBench("01 Original", idx, base, queries, gt, k);
    }
    // ── Completo ─────────────────────────────────────────────────────────────
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgFull();
        runBench("02 Completo M1+M2+M3+M4", idx, base, queries, gt, k);
    }

    std::cout << sep87d << "\n";

    // ── Mejoras individuales ──────────────────────────────────────────────────
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgM1();
        runBench("03 Solo M1  (mL)", idx, base, queries, gt, k);
    }
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgM2();
        runBench("04 Solo M2  (ef_c)", idx, base, queries, gt, k);
    }
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgM3();
        runBench("05 Solo M3  (angular)", idx, base, queries, gt, k);
    }
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgM4();
        runBench("06 Solo M4  (ef_search)", idx, base, queries, gt, k);
    }

    std::cout << sep87d << "\n";

    // ── Combinaciones ─────────────────────────────────────────────────────────
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgM1M4();
        runBench("07 M1+M4    (mL+ef_search)", idx, base, queries, gt, k);
    }
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgM2M4();
        runBench("08 M2+M4    (ef_c+ef_search)", idx, base, queries, gt, k);
    }
    {
        hnswlib::HierarchicalNSWAdaptive<float> idx(&space, n, M, ef_c);
        idx.setEf(ef_s); idx.cfg = cfgM124();
        runBench("09 M1+M2+M4 (sin angular)", idx, base, queries, gt, k);
    }

    std::cout << sep87 << "\n";
}

// =============================================================================
int main(int argc, char** argv) {
    size_t n    = 10000;
    size_t nq   = 500;
    size_t k    = 10;
    size_t M    = 16;
    size_t ef_c = 200;
    size_t ef_s = 50;

    if (argc > 1) n    = std::stoul(argv[1]);
    if (argc > 2) nq   = std::stoul(argv[2]);

    std::cout
        << "================================================================\n"
        << "  HNSW Benchmark v5b — tres escenarios\n"
        << "================================================================\n"
        << "  Elementos    : " << n    << "\n"
        << "  Consultas    : " << nq   << "\n"
        << "  k            : " << k    << "\n"
        << "  M            : " << M    << "\n"
        << "  ef_construct : " << ef_c << "\n"
        << "  ef_search    : " << ef_s << "\n"
        << "================================================================\n";

    // ── ESCENARIO A: Uniforme dim=128 ────────────────────────────────────────
    // Contexto homogeneo: AdaptiveState ~0.5 siempre.
    // Sirve para medir overhead puro del pipeline adaptativo.
    {
        std::cout << "\nGenerando escenario A (uniforme dim=128)... " << std::flush;
        size_t dim = 128;
        auto base    = generateUniform(n,  dim, 42);
        auto queries = generateUniform(nq, dim, 99);
        auto gt      = computeGroundTruth(base, queries, k);
        std::cout << "OK\n";
        runScenario("A: Uniforme dim=128  [contexto homogeneo, M1-M3 inactivas]",
                    base, queries, gt, n, dim, k, M, ef_c, ef_s);
    }

    // ── ESCENARIO B: Clusters dim=128 ────────────────────────────────────────
    // 20 clusters gaussianos, sigma=0.05.
    // density intra-cluster ~67x mayor que uniforme.
    // M1-M3 deben activarse; ef_search sube en zonas inter-cluster.
    {
        std::cout << "\nGenerando escenario B (clusters dim=128)... " << std::flush;
        size_t dim = 128;
        auto base    = generateClustered(n,  dim, 20, 0.05f, 42);
        auto queries = generateClustered(nq, dim, 20, 0.05f, 99);
        auto gt      = computeGroundTruth(base, queries, k);
        std::cout << "OK\n";
        runScenario("B: Clusters dim=128  [densidad variada, M1-M3 activas]",
                    base, queries, gt, n, dim, k, M, ef_c, ef_s);
    }

    // ── ESCENARIO C: Uniforme dim=32 ─────────────────────────────────────────
    // Dimension baja: M3 angular se activa (dim <= angular_max_dim=64).
    // Recall base mas alto, menor maldicion de dimensionalidad.
    {
        std::cout << "\nGenerando escenario C (uniforme dim=32)... " << std::flush;
        size_t dim = 32;
        auto base    = generateUniform(n,  dim, 42);
        auto queries = generateUniform(nq, dim, 99);
        auto gt      = computeGroundTruth(base, queries, k);
        std::cout << "OK\n";
        runScenario("C: Uniforme dim=32   [M3 angular activa]",
                    base, queries, gt, n, dim, k, M, ef_c, ef_s);
    }

    std::cout
        << "\nLeyenda:\n"
        << "  Recall@k     ideal = 1.0\n"
        << "  Nodes/query  distancias computadas por consulta; menor = mas eficiente\n"
        << "  M1  mL adaptativo    disperso -> jerarquia mas profunda\n"
        << "  M2  ef_c adaptativo  denso/heterogeneo -> mas exploracion en construccion\n"
        << "  M3  angular          denso -> filtro estricto (solo dim <= 64)\n"
        << "  M4  ef_search        difficulty alta -> mas presupuesto de busqueda\n\n"
        << "  Escenario A: mide overhead del pipeline (M1-M3 casi inactivas)\n"
        << "  Escenario B: mide ganancia real de M1-M3 con datos con estructura\n"
        << "  Escenario C: mide M3 angular en dimension baja\n\n";

    return 0;
}