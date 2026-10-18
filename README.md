# Modelo de saltos por difusión de Merton (1976)

Universidad Externado de Colombia · Teoría, simulación y calibración en renta variable
(clase de 120 min: 60 de teoría + 60 de simulación).

`merton_externado.py` es un cuaderno [marimo](https://marimo.io) que contiene la presentación completa:
la Hora 1 (teoría, con verificaciones numéricas) y la Hora 2 (simulación, valoración, calibración por MLE,
smile y cobertura). Los controles son interactivos y los valores por defecto son los de las diapositivas.

## Contenido

| Bloque | Qué hay |
|---|---|
| Hora 1 | EDS, solución explícita, martingala, ley condicional, cumulantes, fórmula de valoración, cota de truncamiento, función característica, limitaciones |
| Resultado 1–2 | Simulación exacta, martingala (con casilla para omitir el compensador), momentos |
| Resultado 3–4 | Suma truncada vs. Monte Carlo, error de truncamiento y cota de Poisson |
| Resultado 5–6 | Calibración por MLE, identificación débil de λ (verosimilitud perfilada) |
| Tarea 3 | Calibración con un CSV propio (retornos o precios) |
| Smile | Volatilidad implícita por inversión de Black–Scholes |
| Resultado 7 | Cobertura delta y mercado incompleto |

## Ver el cuaderno sin instalar nada

Si el repositorio tiene GitHub Pages activado (ver abajo), la versión interactiva queda en:

```
https://<usuario>.github.io/<repositorio>/
```

Corre Python dentro del navegador (WebAssembly). La primera carga descarga numpy, scipy y matplotlib
(~30–60 s) y necesita internet. Abre la página antes de la clase.

## Publicar en GitHub Pages

1. Crea un repositorio y sube estos archivos a la rama `main`.
2. En *Settings → Pages*, elige **Source: GitHub Actions**.
3. El flujo `.github/workflows/deploy.yml` exporta el cuaderno y lo publica en cada `push`.

Para ver el código en la versión publicada, cambia `--no-show-code` por `--show-code` en el flujo.

## Correrlo localmente

```bash
pip install -r requirements.txt
marimo edit merton_externado.py      # editar y explorar
marimo run  merton_externado.py      # modo presentación (sin código)
```

Con [uv](https://docs.astral.sh/uv/) no hace falta instalar nada antes:

```bash
uvx marimo edit --sandbox merton_externado.py
```

## Calibrar con datos de ejemplo o propios

En la sección de la Tarea 3 se arrastra un CSV con retornos diarios o precios. Se usa la columna llamada
`ret`, `return`, `close`, `price` (o equivalentes en español) si existe; si no, la última columna numérica.

La carpeta `data/` trae series reales (S&P 500 1950–2018, AAPL, AMZN, NFLX, FB, JPM, XOM) y tres series
sintéticas con parámetros conocidos. Ver `data/README.md` para fuentes y resultados de referencia.
En el navegador hay que descargar el archivo del repositorio y arrastrarlo al cuaderno.

## Notas

- Las secciones pesadas (MLE, perfil de λ, cobertura) tienen un botón para ejecutarlas.
- En el navegador (WebAssembly) las simulaciones grandes tardan más que en Python nativo.
  Si hace falta, baja el número de trayectorias en los menús desplegables.
- `merton_cf` y `carr_madan_call` están en el núcleo del cuaderno como punto de partida para la tarea de extensión (Vía B).
