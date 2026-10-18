# Datos de ejemplo para la calibración (Tarea 3)

Todos los archivos se pueden arrastrar al cuaderno (sección de la Tarea 3).
Los reales tienen columnas `date, close, ret` (`ret` = retorno logarítmico diario).
Los sintéticos tienen `date, ret`. Con **Los datos son: Retornos logarítmicos** se usa `ret`.

## Datos reales

| Archivo | Serie | Periodo | Filas |
|---|---|---|---|
| `sp500_1950_2018.csv` | S&P 500 (índice) | 1950-01-04 a 2018-12-21 | 17 355 |
| `sp500_2000_2018.csv` | S&P 500 (índice) | 2000-01-03 a 2018-12-21 | 4 774 |
| `sp500_2013_2018.csv` | S&P 500 (índice) | 2013-02-11 a 2018-02-07 | 1 258 |
| `aapl_2013_2018.csv` | Apple | 2013-02-11 a 2018-02-07 | 1 258 |
| `amzn_2013_2018.csv`, `nflx_2013_2018.csv`, `fb_2013_2018.csv`, `jpm_2013_2018.csv`, `xom_2013_2018.csv` | Amazon, Netflix, Facebook, JPMorgan, Exxon | 2013-02-11 a 2018-02-07 | 1 258 c/u |
| `aapl_precios_2013_2018.csv` | Apple, **solo precios** (`date, close`) | 2013-02-08 a 2018-02-07 | 1 259 |

`aapl_precios_2013_2018.csv` sirve para probar la opción **Precios** (el cuaderno calcula los log-retornos).
La ventana de `sp500_2013_2018.csv` coincide con la de las acciones, para comparar índice contra acción.
La serie larga incluye el 19-oct-1987 (retorno logarítmico de −22.9 %) y la de 2000–2018 incluye la crisis de 2008.

**Fuentes.**
- S&P 500: archivo `csv/sp500.csv` del repositorio `vijinho/sp500` (datos diarios de Yahoo Finance, cierre).
  Cierre ajustado igual al cierre en esta serie.
- Acciones: `all_stocks_5yr.csv` (dataset de Kaggle *S&P 500 Stock Data*, copia en `plotly/datasets`), cierres diarios.
- No verifiqué las condiciones de redistribución de estas fuentes; son datos públicos de precios, usados aquí con fines docentes.
  Cítalas si los publicas.

## Datos sintéticos (verdad conocida)

| Archivo | Parámetros verdaderos (σ, λ, μ_J, σ_J) | Qué muestra |
|---|---|---|
| `ejemplo_retornos_sinteticos.csv` | 0.18, 5, −0.03, 0.04 | El caso de las diapositivas (6048 días, α = 0.02) |
| `sintetico_lambda_alto.csv` | 0.15, 40, −0.01, 0.015 | Saltos frecuentes y pequeños: λ mejor identificado |
| `sintetico_sin_saltos.csv` | 0.20, 0.05, 0, 0.01 | Casi sin saltos: λ no se identifica |

## Resultados de referencia

Calibración con el cuaderno (α = 0.02, 5 puntos iniciales). Pueden variar un poco con los puntos iniciales.
"cota" = el óptimo quedó en el límite λ = 60; en ese caso los errores estándar de la Hessiana no son válidos.

| Archivo | σ | λ | μ_J | σ_J | Exceso de curtosis |
|---|---|---|---|---|---|
| `sp500_1950_2018` | 0.095 | 60 (cota) | −0.002 | 0.015 | 27.0 |
| `sp500_2000_2018` | 0.112 | 60 (cota) | −0.003 | 0.019 | 8.5 |
| `sp500_2013_2018` | 0.081 | 60 (cota) | −0.003 | 0.011 | 3.3 |
| `aapl_2013_2018` | 0.159 | 60 (cota) | −0.003 | 0.022 | 3.8 |
| `amzn_2013_2018` | 0.204 | 18.2 ± 4.6 | −0.012 | 0.050 | 11.0 |
| `nflx_2013_2018` | 0.268 | 41.1 ± 18.2 | −0.002 | 0.051 | 12.9 |
| `fb_2013_2018` | 0.193 | 60 (cota) | −0.002 | 0.029 | 27.7 |
| `jpm_2013_2018` | 0.140 | 60 (cota) | −0.003 | 0.019 | 3.3 |
| `xom_2013_2018` | 0.117 | 60 (cota) | −0.001 | 0.017 | 3.5 |
| `ejemplo_retornos_sinteticos` | 0.176 | 5.67 ± 0.87 | −0.028 | 0.039 | 10.7 |
| `sintetico_lambda_alto` | 0.150 | 34.0 ± 6.4 | −0.011 | 0.016 | 2.7 |
| `sintetico_sin_saltos` | 0.199 | 14.9 ± 45.7 | +0.007 | 0.005 | 0.0 |

**Lectura para la clase.** Con datos reales diarios la mezcla suele usar muchos saltos diminutos (λ en la cota,
σ_J pequeño) para imitar las colas pesadas, no pocos saltos grandes. Amazon y Netflix, con saltos por resultados,
dan óptimos interiores. En `sintetico_sin_saltos` el error estándar de λ es enorme: el parámetro no se identifica.

## Tiempo de cálculo

`sp500_1950_2018.csv` tarda ~14 s en Python nativo; en el navegador (WebAssembly) puede tardar bastante más.
Para la clase conviene empezar con `sp500_2013_2018.csv` o `aapl_2013_2018.csv` (menos de 1 s en nativo).
