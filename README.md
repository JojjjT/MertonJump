# Modelo de Merton (1976) · Teoría y simulación

Material interactivo de la clase sobre el **modelo de saltos por difusión de Merton (1976)** para la Universidad Externado de Colombia.

## 🔗 Enlaces

| Recurso | URL |
|---|---|
| 📓 Notebook interactivo (marimo, WebAssembly) | https://jojjjt.github.io/MertonJump/ |
| 🎞️ Presentación (Marp, HTML estático) | https://jojjjt.github.io/MertonJump/merton_externado_marp.html |

## 📂 Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `merton_externado.py` | Notebook marimo con teoría, simulaciones y calibración por MLE |
| `merton_externado_marp.html` | Presentación Marp exportada a HTML (autocontenida) |
| `.github/workflows/deploy.yml` | Workflow único que publica ambos recursos en GitHub Pages |

## 🚀 Cómo se publica

Un **único workflow** (`.github/workflows/deploy.yml`) se dispara con cambios en cualquiera de los dos archivos y construye el sitio completo:

1. Exporta `merton_externado.py` a WebAssembly con `marimo export html-wasm` (instalando `uv` para resolver las dependencias inline).
2. Copia `merton_externado_marp.html` tal cual a `_site/`.
3. Sube el artifact y despliega en GitHub Pages.

Esto garantiza que **ambos recursos estén siempre accesibles al mismo tiempo**, sin que un despliegue sobrescriba al otro.

## 🛠️ Ejecución local

### Notebook marimo

```bash
pip install uv marimo
marimo edit merton_externado.py
```

### Exportar a WebAssembly

```bash
marimo export html-wasm merton_externado.py -o _site --mode run --no-show-code
python -m http.server --directory _site
```

Luego abre `http://localhost:8000` en el navegador.

### Presentación Marp

Abre `merton_externado_marp.html` directamente en cualquier navegador, o sírvelo con:

```bash
python -m http.server 8001
```

## 📚 Referencias

- Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous*. Journal of Financial Economics, 3(1–2), 125–144.
- [Documentación de marimo](https://docs.marimo.io/)
- [marimo WebAssembly export](https://docs.marimo.io/guides/wasm/)
- [Marp](https://marp.app/)
