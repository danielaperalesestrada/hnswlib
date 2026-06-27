# Chroma-HNSWlib Adaptativo

Fork modificado de [chroma-hnswlib](https://github.com/chroma-core/hnswlib) desarrollado como trabajo para el curso de **Estructuras de Datos Avanzadas**.

## Objetivo

Mejorar el **recall** del algoritmo HNSW (Hierarchical Navigable Small World) mediante una estructura adaptativa que ajusta dinámicamente sus parámetros internos según las características locales de los datos durante la construcción y búsqueda del índice.

## ¿Qué es HNSW?

HNSW es un algoritmo de búsqueda aproximada de vecinos más cercanos (ANN) que organiza los vectores en un grafo jerárquico de múltiples capas. Es ampliamente usado en bases de datos vectoriales como [Chroma](https://github.com/chroma-core/chroma) por su velocidad y precisión.

El problema con la implementación original es que usa parámetros **fijos** durante toda la construcción del índice, sin considerar la estructura local de los datos en cada momento.

## Modificación propuesta: HierarchicalNSWAdaptive

Se implementó la clase `HierarchicalNSWAdaptive` (en `hnswlib/hnswalg_adaptive.h`) que hereda de `HierarchicalNSW` y sobreescribe los métodos `addPoint` y `searchKnn` con comportamiento adaptativo.

### Mejoras implementadas (M1-M4)

| Mejora | Descripción |
|--------|-------------|
| **M1 — mL adaptativo** | Ajusta la escala de niveles según la densidad local. Zonas dispersas generan jerarquías más profundas |
| **M2 — ef_construction adaptativo** | Aumenta el presupuesto de construcción en zonas densas o heterogéneas |
| **M3 — Diversificación angular** | En dimensiones bajas (≤64), filtra vecinos demasiado similares angularmente para mejorar la cobertura |
| **M4 — ef_search adaptativo** | Ajusta el presupuesto de búsqueda según la dificultad estimada de la consulta |

### AdaptiveState

Cada inserción y búsqueda calcula un `AdaptiveState` con tres métricas locales:

- **density_score** — qué tan densa es la zona local (0=dispersa, 1=densa)
- **variance_score** — heterogeneidad de las distancias locales
- **skewness_score** — asimetría de la distribución de distancias

Estas métricas se combinan en un **difficulty score** que guía las decisiones adaptativas.

## Cambios en el código

### `hnswlib/hnswalg.h`
- Se agregó `virtual` a `addPoint` y `searchKnn` para permitir polimorfismo con la clase derivada

### `hnswlib/hnswalg_adaptive.h` *(archivo nuevo)*
- Implementación completa de `HierarchicalNSWAdaptive`

### `python_bindings/bindings.cpp`
- `init_new_index` ahora instancia `HierarchicalNSWAdaptive` en lugar de `HierarchicalNSW`

### `src/bindings.cpp`
- Mismo cambio para la interfaz Rust/C usada por Chroma

## Integración con Chroma

Esta versión modificada es compatible con **Chroma 0.5.20**. Para usarla:

```bash
# 1. Clonar este repo
git clone https://github.com/tu-usuario/hnswlib.git
cd hnswlib

# 2. Crear entorno virtual (usar x64 Native Tools Command Prompt en Windows)
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install numpy pybind11 setuptools wheel

# 4. Compilar e instalar
python setup.py build_ext --inplace
pip install -e .

# 5. Instalar Chroma (sin sobreescribir tu hnswlib)
pip install chromadb==0.5.20 --no-deps
pip install onnxruntime posthog opentelemetry-instrumentation-fastapi
```

> **Importante:** Ejecutar siempre desde la terminal **x64 Native Tools Command Prompt for VS 2022** en Windows para que el compilador C++ sea de 64 bits.

> **Importante:** No ejecutar Python desde la carpeta raíz del repo para evitar conflictos con el módulo Python.

## Resultados

*Pendiente — se añadirán los resultados de benchmarks comparando HierarchicalNSW original vs HierarchicalNSWAdaptive en términos de recall, tiempo de inserción y tiempo de búsqueda.*

## Estructura del proyecto

```
hnswlib/
├── hnswlib/
│   ├── hnswalg.h              # Algoritmo HNSW original (modificado: virtual)
│   ├── hnswalg_adaptive.h     # ← Nueva clase adaptativa
│   └── ...
├── python_bindings/
│   └── bindings.cpp           # Bindings Python (modificado)
├── src/
│   └── bindings.cpp           # Bindings Rust/C (modificado)
├── benchmarks/
│   └── bench_comparison.cpp   # Benchmark C++ (escenarios A, B, C)
└── benchmarks_chroma/         # ← Benchmarks con Chroma (nuevo)
```

## Autor

Daniela Perales — Curso de Estructuras de Datos Avanzadas