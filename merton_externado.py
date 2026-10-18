# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25",
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Merton (1976) · Teoría y simulación")


@app.cell(hide_code=True)
def _():
    import csv
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import simpson
    from scipy.optimize import brentq, minimize
    from scipy.special import gammaln, logsumexp, ndtr
    from scipy.stats import norm, poisson

    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.2),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return (
        brentq,
        csv,
        gammaln,
        logsumexp,
        math,
        minimize,
        mo,
        ndtr,
        norm,
        np,
        plt,
        poisson,
        simpson,
    )


# ======================================================================
# Portada, objetivos, ruta, guía de uso
# ======================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **UNIVERSIDAD EXTERNADO DE COLOMBIA**

    # Modelo de saltos por difusión de Merton (1976)

    ## Teoría, simulación y calibración en renta variable

    Acciones e índices bursátiles · Clase de 120 min (60 teoría + 60 simulación)

    *Este cuaderno contiene la presentación completa. Cada bloque de teoría va seguido de la
    **celda de código** que lo implementa y de un **laboratorio interactivo**: mueve los
    deslizadores y el gráfico se recalcula. Los valores por defecto son los de las diapositivas.*

    > **Cómo proyectarlo en clase.**
    > `marimo edit merton_marimo.py` → se ve el código de todas las celdas (modo edición).
    > `marimo run --include-code merton_marimo.py` → modo aplicación, pero mostrando el código
    > debajo de cada resultado. Las celdas de deslizadores están ocultas para no distraer; se
    > despliegan con el icono del ojo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Objetivos de aprendizaje

    Al terminar la sesión, el estudiante podrá:

    1. **Especificar** la EDS de Merton, resolverla con Itô con saltos y **demostrar** la martingala que impone el compensador $\lambda m$.
    2. **Derivar** el precio de una opción europea como mezcla Poisson de precios Black–Scholes ($r_k$, $\sigma_k$).
    3. **Acotar** el error de truncamiento con una cota explícita de cola de Poisson y **verificar** contra Monte Carlo.
    4. **Calibrar** $(\sigma,\lambda,\mu_J,\sigma_J)$ por MLE y **diagnosticar** identificación débil.
    5. **Explicar** el mercado incompleto, el fallo del delta-hedge y los límites del smile.

    ---
    # Ruta de la sesión · 120 minutos

    | Hora | Bloque | Min | Contenido | Código / laboratorio |
    |---|---|---|---|---|
    | **1 · Teoría** | 0 | 0–12 | Contexto histórico y motivación empírica | — |
    | | 1 | 12–24 | EDS, solución explícita, martingala | `simulate_decomposed` · **Lab 1** |
    | | 2 | 24–36 | Ley condicional, cumulantes | **Lab 2** (mezcla) · gráfico de cumulantes |
    | | 3 | 36–49 | **Fórmula de valoración** y cota de truncamiento | `merton_terms` · **Lab 3** (términos) · **Lab 4** (sensibilidad) · **Lab 5** (mapa) |
    | | 4 | 49–57 | Función característica y smile | `carr_madan_call` |
    | | 5 | 57–60 | Limitaciones | — |
    | **2 · Simulación** | 1–2 | 0–26 | Simulación exacta; suma truncada vs. Monte Carlo | `simulate_merton` · **Lab 6** (martingala y Poisson) · **Lab 7** (colas) |
    | | 3 | 26–40 | **Calibración por MLE** | `nll`, `fit` · **Lab 8** (superficie) · **Lab 9** (Monte Carlo de estimadores) · **Lab 10** (detección de saltos) |
    | | 4 | 40–52 | Smile e inversión de Black–Scholes | `merton_iv` · **Lab 11** (estructura temporal del skew) |
    | | 5 | 52–60 | Cobertura, mercado incompleto, ticket de salida | `delta_hedge_pnl` · **Lab 12** (frecuencia de rebalanceo) |
    """)
    return


# ======================================================================
#  HORA 1 · TEORÍA
# ======================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **HORA 1**

    # Teoría

    ---
    # Contexto histórico

    - **1973 · Black–Scholes–Merton.** Browniano geométrico: trayectorias continuas, retornos log-normales, volatilidad constante.
    - **Evidencia de discontinuidades.** Crash de octubre de 1987 (≈ −20 % en un día), anuncios de resultados, política monetaria, eventos geopolíticos.
    - **1976 · Merton**, *Option pricing when underlying stock returns are discontinuous* (JFE): difusión + Poisson compuesto de saltos multiplicativos.
    - **Después:** Bates (1996), Duffie–Pan–Singleton (2000), Kou (2002).

    > **Pregunta guía.** ¿Cómo construimos, implementamos y calibramos un modelo que capture saltos en acciones e índices, y qué precio pagamos en rigor (completitud, identificación)?

    ---
    # Motivación empírica en renta variable

    - **Exceso de curtosis** muy superior a 0; eventos de más de $5\sigma$ inconsistentes con la normal (2008, COVID-2020).
    - **Asimetría negativa** en índices; saltos idiosincráticos en acciones por resultados.
    - **Superficies de volatilidad implícita** (SPX, SX5E): skew negativo persistente, más fuerte a plazos cortos.

    > **Verificación (2 min, parejas).** (i) ¿Qué supuesto de BS se rompe cuando una acción salta tras un anuncio? (ii) ¿Qué ley describe el número de saltos en $[0,T]$? (iii) ¿Qué parámetro fija la frecuencia esperada de saltos?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Marco probabilístico y supuestos

    Espacio filtrado $(\Omega,\mathcal F,(\mathcal F_t),\mathbb Q)$ con tres objetos **mutuamente independientes**:

    - $W$: $\mathbb Q$-browniano estándar, $\sigma>0$.
    - $N$: Poisson homogéneo de intensidad $\lambda>0$, tiempos de salto $\tau_i$.
    - $J_i=\ln Y_i\sim\mathcal N(\mu_J,\sigma_J^2)$ i.i.d., independientes de $(W,N)$.

    Compensador: $m:=\mathbb E[Y-1]=e^{\mu_J+\sigma_J^2/2}-1$.

    > **Supuestos que no son teoremas:** independencia de $W,N,J$; saltos lognormales; $\mathbb Q$ **exógena**. Merton (1976) supone riesgo de salto *no sistemático*, por lo que no exige prima de riesgo de salto.

    ---
    # EDS y estructura de martingala

    $$\frac{dS_t}{S_{t^-}}=(r-\lambda m)\,dt+\sigma\,dW_t+(Y-1)\,dN_t$$

    Con $M_t:=\sum_{i=1}^{N_t}(Y_i-1)-\lambda m\,t$, una $\mathbb Q$-martingala:

    $$\frac{dS_t}{S_{t^-}}=r\,dt+\sigma\,dW_t+dM_t\;\Longrightarrow\;e^{-rt}S_t=S_0\,\mathcal E(\sigma W+M)_t$$

    ($\mathcal E$: exponencial de Doléans-Dade; $\Delta M=Y-1>-1$ ⇒ estrictamente positivo.)

    - $S_{t^-}$ es el límite por la izquierda: integrando **predecible**.
    - Sin $-\lambda m\,dt$, $M$ no sería martingala y $\mathbb Q$ no sería neutral al riesgo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Solución explícita (Itô con saltos)

    Entre saltos: $d\ln S=(r-\lambda m-\tfrac{\sigma^2}{2})dt+\sigma\,dW$. **En un salto**:

    $$\Delta\ln S_{\tau_i}=\ln\!\Big(\frac{S_{\tau_i^-}Y_i}{S_{\tau_i^-}}\Big)=J_i .$$

    $$S_t=S_0\exp\!\Big[\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)t+\sigma W_t+\sum_{i=1}^{N_t}J_i\Big]$$

    > $\ln S_t$ es un **proceso de Lévy** (browniano con deriva + Poisson compuesto) con medida de Lévy finita $\nu(dx)=\lambda f_J(x)\,dx$: actividad finita, $\nu(\mathbb R)=\lambda<\infty$.

    **Del papel al código.** La solución explícita es una suma de tres piezas: *deriva compensada*,
    *difusión* y *saltos acumulados*. La función `simulate_decomposed` devuelve cada pieza por
    separado para que se vea qué aporta cada una.
    """)
    return


@app.cell
def _(math, np):
    def comp_m(muJ, sigJ):
        """Compensador m = E[Y - 1] = exp(muJ + sigJ^2/2) - 1."""
        return math.exp(muJ + 0.5 * sigJ**2) - 1.0

    def simulate_decomposed(S0, r, sigma, lam, muJ, sigJ, T, n, seed=0):
        """Una trayectoria de ln S separada en deriva, difusión y saltos."""
        rng = np.random.default_rng(seed)
        dt = T / n
        m = comp_m(muJ, sigJ)
        t = np.linspace(0.0, T, n + 1)

        dN = rng.poisson(lam * dt, n)                      # nº de saltos por paso
        J = rng.normal(muJ * dN, sigJ * np.sqrt(dN))       # suma de dN saltos ~ N(dN·μ_J, dN·σ_J²)
        dW = rng.normal(0.0, math.sqrt(dt), n)             # incrementos brownianos

        drift = (r - lam * m - 0.5 * sigma**2) * t         # deriva compensada
        diff = np.concatenate([[0.0], np.cumsum(sigma * dW)])
        jumps = np.concatenate([[0.0], np.cumsum(J)])
        lnS = np.log(S0) + drift + diff + jumps
        return t, lnS, drift, diff, jumps, dN

    return comp_m, simulate_decomposed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 1 · Descomposición de una trayectoria

    Cambia la **semilla** para ver otra realización. Con $\lambda=0$ solo queda la difusión (BS).
    Observa que el salto es **instantáneo** en $S_t$ y que cada salto es una variable
    $J_i\sim\mathcal N(\mu_J,\sigma_J^2)$ en el log-precio.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    l1_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    l1_lam = mo.ui.slider(0.0, 20.0, step=0.5, value=5.0, label="λ")
    l1_muJ = mo.ui.slider(-0.4, 0.2, step=0.01, value=-0.10, label="μ_J")
    l1_sigJ = mo.ui.slider(0.01, 0.5, step=0.01, value=0.20, label="σ_J")
    l1_T = mo.ui.slider(0.25, 5.0, step=0.25, value=1.0, label="T")
    l1_seed = mo.ui.slider(0, 200, step=1, value=3, label="semilla")
    mo.vstack(
        [
            mo.hstack([l1_sigma, l1_lam, l1_T], justify="start"),
            mo.hstack([l1_muJ, l1_sigJ, l1_seed], justify="start"),
        ]
    )
    return l1_T, l1_lam, l1_muJ, l1_seed, l1_sigJ, l1_sigma


@app.cell
def _(l1_T, l1_lam, l1_muJ, l1_seed, l1_sigJ, l1_sigma, mo, np, plt, simulate_decomposed):
    _S0, _r = 100.0, 0.05
    _n = int(252 * l1_T.value)
    _t, _lnS, _drift, _diff, _jumps, _dN = simulate_decomposed(
        _S0, _r, l1_sigma.value, l1_lam.value, l1_muJ.value, l1_sigJ.value,
        l1_T.value, _n, seed=l1_seed.value,
    )
    _S = np.exp(_lnS)
    _idx = np.where(_dN > 0)[0] + 1  # índices de los días con salto

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8))
    _axs[0].plot(_t, _S, lw=1.2)
    _axs[0].scatter(_t[_idx], _S[_idx], c="k", s=18, zorder=3, label="días con salto")
    _axs[0].plot(_t, _S0 * np.exp(_r * _t), "k--", lw=1, label=r"$S_0e^{rt}$")
    _axs[0].set_title("Precio $S_t$")
    _axs[0].set_xlabel("t")
    _axs[0].legend()

    _axs[1].plot(_t, _lnS - np.log(_S0), "k", lw=1.4, label=r"$\ln(S_t/S_0)$ total")
    _axs[1].plot(_t, _drift + _diff, lw=1, label="deriva compensada + difusión")
    _axs[1].plot(_t, _jumps, lw=1.4, label=r"saltos acumulados $\sum J_i$")
    _axs[1].set_title("Las tres piezas de la solución explícita")
    _axs[1].set_xlabel("t")
    _axs[1].legend(fontsize=8)
    _fig.tight_layout()

    _J = np.diff(_jumps)[_idx - 1]
    _lines = "\n".join(
        f"| {_t[i]:.3f} | {j:+.3f} | {100 * (np.exp(j) - 1):+.1f} % |"
        for i, j in list(zip(_idx, _J))[:8]
    )
    _tbl = mo.md(
        f"""
    **{len(_idx)} días con salto** en {l1_T.value} años (esperado: {l1_lam.value * l1_T.value:.1f}).
    Los primeros saltos:

    | t | J = Δln S | variación de S |
    |---|---|---|
    {_lines}
    """
    )
    mo.vstack([_fig, _tbl])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Demostración de la martingala

    Por independencia, $\mathbb E[S_t]$ se factoriza:

    $$\mathbb E[S_t]=S_0\,e^{(r-\lambda m)t}\;\underbrace{\mathbb E\big[e^{\sigma W_t-\sigma^2t/2}\big]}_{=1}\;\underbrace{\mathbb E\Big[\prod_{i=1}^{N_t}Y_i\Big]}_{=\,e^{\lambda m t}}$$

    Condicionando en $N_t$:

    $$\mathbb E\Big[\prod_{i=1}^{N_t}Y_i\Big]=\sum_{k\ge0}e^{-\lambda t}\frac{(\lambda t)^k}{k!}(1+m)^k=e^{\lambda t\,m}\;\Rightarrow\;\boxed{\mathbb E[S_t]=S_0e^{rt}}$$

    y con el mismo argumento $\mathbb E[S_t\mid\mathcal F_s]=S_se^{r(t-s)}$.

    > **Pausa (2 min).** ¿Qué pasa con $\mathbb E[S_t]$ si se omite $\lambda m$? (Crece a tasa $r+\lambda m$.)
    > Compruébalo en el **Laboratorio 6** (Hora 2): compara la media simulada con y sin compensador.

    ---
    # Ley condicional: mezcla Poisson de normales

    Dado $N_T=k$, la suma de $k$ saltos es $\mathcal N(k\mu_J,k\sigma_J^2)$, independiente de $W_T$:

    $$\ln\frac{S_T}{S_0}\Big|N_T=k\sim\mathcal N(\mu_k,s_k^2),\quad \mu_k=\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)T+k\mu_J,\ \ s_k^2=\sigma^2T+k\sigma_J^2$$

    $$f_X(x)=\sum_{k=0}^{\infty}e^{-\lambda T}\frac{(\lambda T)^k}{k!}\,\varphi(x;\mu_k,s_k^2)$$

    Además $\mathbb E[S_T\mid N_T=k]=S_0e^{r_kT}$ con

    $$r_k=r-\lambda m+\frac{k}{T}\Big(\mu_J+\frac{\sigma_J^2}{2}\Big),\qquad \sigma_k^2=\sigma^2+\frac{k\sigma_J^2}{T}.$$

    ## Laboratorio 2 · La mezcla Poisson, componente por componente

    A la izquierda, cada campana coloreada es $P_k\,\varphi(x;\mu_k,s_k^2)$; la curva negra es su suma.
    A la derecha, los pesos $P_k$. Sube $\lambda T$: la mezcla se mueve hacia la normal (TCL).
    Baja $\lambda T$ a 0.3: casi toda la masa está en $k=0$ y las colas vienen de $k=1,2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    m_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    m_lam = mo.ui.slider(0.1, 10.0, step=0.1, value=1.0, label="λ")
    m_T = mo.ui.slider(0.25, 5.0, step=0.25, value=1.0, label="T")
    m_muJ = mo.ui.slider(-0.4, 0.2, step=0.01, value=-0.20, label="μ_J")
    m_sigJ = mo.ui.slider(0.01, 0.5, step=0.01, value=0.15, label="σ_J")
    mo.vstack(
        [
            mo.hstack([m_sigma, m_lam, m_T], justify="start"),
            mo.hstack([m_muJ, m_sigJ], justify="start"),
        ]
    )
    return m_T, m_lam, m_muJ, m_sigJ, m_sigma


@app.cell
def _(comp_m, m_T, m_lam, m_muJ, m_sigJ, m_sigma, norm, np, plt, poisson):
    _r = 0.05
    _sg, _lam, _mu, _sj, _T = m_sigma.value, m_lam.value, m_muJ.value, m_sigJ.value, m_T.value
    _m = comp_m(_mu, _sj)

    _k = np.arange(0, 25)
    _w = poisson.pmf(_k, _lam * _T)                       # P_k
    _mu_k = (_r - _lam * _m - 0.5 * _sg**2) * _T + _k * _mu
    _s_k = np.sqrt(_sg**2 * _T + _k * _sj**2)

    _mean = (_w * _mu_k).sum()
    _var = (_w * (_s_k**2 + _mu_k**2)).sum() - _mean**2
    _grid = np.linspace(_mean - 5 * np.sqrt(_var), _mean + 5 * np.sqrt(_var), 900)
    _comp = _w[:, None] * norm.pdf(_grid[None, :], _mu_k[:, None], _s_k[:, None])

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    for _i in range(6):
        _axs[0].fill_between(_grid, _comp[_i], alpha=0.35, label=f"k = {_i}")
    _axs[0].plot(_grid, _comp.sum(0), "k", lw=1.6, label="mezcla")
    _axs[0].plot(_grid, norm.pdf(_grid, _mean, np.sqrt(_var)), "r--", lw=1.2, label="normal (BS)")
    _axs[0].set_xlabel(r"$\ln(S_T/S_0)$")
    _axs[0].set_title("Densidad de ln(S_T/S_0)")
    _axs[0].legend(fontsize=8)

    _axs[1].bar(_k[:12], _w[:12], color="C0")
    _axs[1].set_xlabel("k (número de saltos)")
    _axs[1].set_title(rf"Pesos $P_k$ · $\lambda T$ = {_lam * _T:.2f}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Cumulantes: derivación y lectura

    $$K_X(u)=u\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)T+\tfrac{\sigma^2u^2}{2}T+\lambda T\big(e^{u\mu_J+u^2\sigma_J^2/2}-1\big)$$

    Con $\kappa_n=\lambda T\,\mathbb E[J^n]$ para $n\ge3$:

    $$\kappa_2=\sigma^2T+\lambda T(\sigma_J^2+\mu_J^2),\quad \kappa_3=\lambda T(\mu_J^3+3\mu_J\sigma_J^2),\quad \kappa_4=\lambda T(\mu_J^4+6\mu_J^2\sigma_J^2+3\sigma_J^4)$$

    $$\gamma_1=\frac{\kappa_3}{\kappa_2^{3/2}}=O(T^{-1/2}),\qquad \gamma_2=\frac{\kappa_4}{\kappa_2^2}=O(T^{-1})$$

    > Con $\mu_J<0$: $\gamma_1<0$ y $\gamma_2>0$. Con $\lambda=0$ los cumulantes de orden $\ge3$ se anulan (BS). La normalidad se recupera al crecer $T$.

    **Verificación numérica:** el gráfico muestra $|\gamma_1|$ y $\gamma_2$ contra $T$ en escala log-log, con las pendientes teóricas $-1/2$ y $-1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    c_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    c_lam = mo.ui.slider(0.1, 20.0, step=0.1, value=5.0, label="λ")
    c_muJ = mo.ui.slider(-0.4, 0.2, step=0.01, value=-0.10, label="μ_J")
    c_sigJ = mo.ui.slider(0.01, 0.5, step=0.01, value=0.20, label="σ_J")
    mo.hstack([c_sigma, c_lam, c_muJ, c_sigJ], justify="start")
    return c_lam, c_muJ, c_sigJ, c_sigma


@app.cell
def _(c_lam, c_muJ, c_sigJ, c_sigma, np, plt):
    _sg, _lam, _mu, _sj = c_sigma.value, c_lam.value, c_muJ.value, c_sigJ.value
    _T = np.logspace(-2, 2, 120)
    _k2 = _sg**2 * _T + _lam * _T * (_sj**2 + _mu**2)
    _k3 = _lam * _T * (_mu**3 + 3 * _mu * _sj**2)
    _k4 = _lam * _T * (_mu**4 + 6 * _mu**2 * _sj**2 + 3 * _sj**4)
    _g1, _g2 = _k3 / _k2**1.5, _k4 / _k2**2

    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.loglog(_T, np.abs(_g1), label=r"$|\gamma_1|$")
    _ax.loglog(_T, _g2, label=r"$\gamma_2$")
    _i = 60
    _ax.loglog(_T[_i:], np.abs(_g1[_i]) * (_T[_i:] / _T[_i]) ** -0.5, "k:", lw=1, label=r"pendiente $-1/2$")
    _ax.loglog(_T[_i:], _g2[_i] * (_T[_i:] / _T[_i]) ** -1.0, "k--", lw=1, label=r"pendiente $-1$")
    _ax.set_xlabel("T (años)")
    _ax.set_title("Asimetría y curtosis en exceso de ln(S_T/S_0)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Valoración: derivación en tres pasos

    **Paso 1 · Condicionamiento** ($N$ independiente de $(W,J)$):

    $$C_0=e^{-rT}\sum_{k\ge0}P_k\,\mathbb E^{\mathbb Q}\big[(S_T-K)^+\mid N_T=k\big],\qquad P_k=e^{-\lambda T}\frac{(\lambda T)^k}{k!}$$

    **Paso 2 · Lema lognormal.** Si $\ln S\sim\mathcal N(\mu,s^2)$: $\ \mathbb E[(S-K)^+]=e^{\mu+s^2/2}\Phi(d_1)-K\Phi(d_2)$, $\ d_1=\frac{\mu-\ln K+s^2}{s}$.

    Con $e^{\mu_k+s_k^2/2}=S_0e^{r_kT}$:

    $$d_1^k=\frac{\ln(S_0/K)+(r_k+\sigma_k^2/2)T}{\sigma_k\sqrt T},\qquad d_2^k=d_1^k-\sigma_k\sqrt T$$

    ---
    # Fórmula de Merton

    $$\boxed{C_0=\sum_{k=0}^{\infty}P_k\Big[S_0e^{(r_k-r)T}\Phi(d_1^k)-Ke^{-rT}\Phi(d_2^k)\Big]}$$

    Como $\mathrm{BS}(\cdot,r_k,\sigma_k)=S_0\Phi(d_1^k)-Ke^{-r_kT}\Phi(d_2^k)$: $\ C_0=\sum_kP_k\,e^{(r_k-r)T}\mathrm{BS}(\cdot,r_k,\sigma_k)$.

    Con $\lambda':=\lambda(1+m)$ y $e^{\mu_J+\sigma_J^2/2}=1+m$:

    $$P_ke^{(r_k-r)T}=e^{-\lambda'T}\frac{(\lambda'T)^k}{k!}=:P_k'\;\Rightarrow\;\boxed{C_0=\sum_{k\ge0}P_k'\,\mathrm{BS}(S_0,K,T,r_k,\sigma_k)}$$

    > **No** es $\sum_kP_k\,\mathrm{BS}$ con intensidad $\lambda$: hay que usar $P_k'$ (intensidad $\lambda'$) o el factor $e^{(r_k-r)T}$. Ambas formas son idénticas.

    ---
    # Verificaciones y cota de truncamiento

    - **$\lambda=0$:** $P_0=1$, $r_0=r$, $\sigma_0=\sigma$ → BS exacto.
    - **$K\to0$:** $d_1^k,d_2^k\to\infty$ y $C_0\to S_0\sum_kP'_k=S_0$.
    - **Paridad put–call:** $P_0=C_0-S_0+Ke^{-rT}$ (consecuencia de la martingala).

    **Cota explícita.** Cada $\mathrm{BS}\le S_0$, luego:

    $$0\le C_0-C^{(K^*)}\le S_0\sum_{k>K^*}P_k'=S_0\,\mathbb P(N'>K^*),\qquad N'\sim\text{Poisson}(\lambda'T)$$

    Chernoff, $K^*>\lambda'T$: $\ \mathbb P(N'>K^*)\le e^{-\lambda'T}\big(e\lambda'T/K^*\big)^{K^*}$. Se elige $K^*$ con $S_0\,\mathbb P(N'>K^*)<\varepsilon$.

    > **Pausa (60 s).** Escribe por qué la fórmula colapsa a BS si $\lambda=0$.

    **Del papel al código.** `merton_terms` evalúa $\sum_kP_k'\,\mathrm{BS}(r_k,\sigma_k)$ **vectorizado** (sobre $k$ y sobre $S_0$);
    `merton_breakdown` devuelve los términos uno a uno para poder inspeccionarlos.
    """)
    return


@app.cell
def _(brentq, comp_m, gammaln, math, ndtr, np, poisson):
    def bs_call(S0, K, T, r, sigma):
        d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S0 * ndtr(d1) - K * math.exp(-r * T) * ndtr(d2)

    def bs_put(S0, K, T, r, sigma):
        return bs_call(S0, K, T, r, sigma) - S0 + K * math.exp(-r * T)

    def merton_terms(S0, K, T, r, sigma, lam, muJ, sigJ, kmax=None):
        """Call, delta, lambda' y put de Merton como mezcla Poisson de BS.

        C0 = sum_k P'_k BS(S0, K, T, r_k, sigma_k),   P'_k ~ Poisson(lambda' T).
        kmax=None elige el truncamiento automáticamente; un entero lo fija.
        Vectorizado sobre S0.
        """
        S0 = np.asarray(S0, dtype=float)
        m = comp_m(muJ, sigJ)
        lamp = max(lam * (1 + m), 1e-12)
        if kmax is None:
            kmax = int(lamp * T + 8 * math.sqrt(lamp * T) + 25)
        k = np.arange(kmax + 1)
        Pk = np.exp(-lamp * T + k * np.log(lamp * T) - gammaln(k + 1))   # P'_k
        rk = r - lam * m + k * (muJ + 0.5 * sigJ**2) / T
        sk = np.sqrt(sigma**2 + k * sigJ**2 / T)
        S = S0[..., None]
        sq = math.sqrt(T)
        d1 = (np.log(S / K) + (rk + 0.5 * sk**2) * T) / (sk * sq)
        d2 = d1 - sk * sq
        disc_k = K * np.exp(-rk * T)
        call = (Pk * (S * ndtr(d1) - disc_k * ndtr(d2))).sum(-1)
        put = (Pk * (disc_k * ndtr(-d2) - S * ndtr(-d1))).sum(-1)
        delta = (Pk * ndtr(d1)).sum(-1)
        return call, delta, lamp, put

    def merton_call(S0, K, T, r, sigma, lam, muJ, sigJ, kmax=None):
        return float(merton_terms(np.array(S0), K, T, r, sigma, lam, muJ, sigJ, kmax)[0])

    def merton_put(S0, K, T, r, sigma, lam, muJ, sigJ, kmax=None):
        return float(merton_terms(np.array(S0), K, T, r, sigma, lam, muJ, sigJ, kmax)[3])

    def merton_breakdown(S0, K, T, r, sigma, lam, muJ, sigJ, kmax=15):
        """Términos de la suma: k, P'_k, r_k, sigma_k y BS(S0,K,T,r_k,sigma_k)."""
        m = comp_m(muJ, sigJ)
        lamp = max(lam * (1 + m), 1e-12)
        k = np.arange(kmax + 1)
        Pk = poisson.pmf(k, lamp * T)
        rk = r - lam * m + k * (muJ + 0.5 * sigJ**2) / T
        sk = np.sqrt(sigma**2 + k * sigJ**2 / T)
        bs = np.array([bs_call(S0, K, T, a, b) for a, b in zip(rk, sk)])
        return k, Pk, rk, sk, bs

    def merton_iv(S0, K, T, r, sigma, lam, muJ, sigJ):
        """Volatilidad implícita de BS usando la opción OTM (put si K < S0)."""
        c = merton_call(S0, K, T, r, sigma, lam, muJ, sigJ)
        if K >= S0:
            f = lambda s: bs_call(S0, K, T, r, s) - c
        else:
            p = c - S0 + K * math.exp(-r * T)
            f = lambda s: bs_put(S0, K, T, r, s) - p
        try:
            return brentq(f, 1e-4, 5.0)
        except ValueError:
            return float("nan")

    return (
        bs_call,
        bs_put,
        merton_breakdown,
        merton_call,
        merton_iv,
        merton_put,
        merton_terms,
    )


@app.cell(hide_code=True)
def _(bs_call, math, merton_call, merton_put, mo):
    # Parámetros de la Tarea 2
    _p = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, lam=1.0, muJ=-0.2, sigJ=0.3)
    _c = merton_call(**_p)
    _put = merton_put(**_p)
    _parity = _put - (_c - _p["S0"] + _p["K"] * math.exp(-_p["r"] * _p["T"]))
    _q = dict(_p)
    _q["lam"] = 0.0
    _bs_diff = merton_call(**_q) - bs_call(_p["S0"], _p["K"], _p["T"], _p["r"], _p["sigma"])
    _k0 = dict(_p)
    _k0["K"] = 1e-6
    _lim = merton_call(**_k0) / _p["S0"]
    mo.md(
        rf"""
    **Comprobación numérica** ($S_0=K=100$, $r=0.05$, $\sigma=0.2$, $T=1$, $\lambda=1$, $\mu_J=-0.2$, $\sigma_J=0.3$)

    | Propiedad | Resultado |
    |---|---|
    | $\lambda=0$: $C_0^{{\text{{Merton}}}}-C^{{\text{{BS}}}}$ | {_bs_diff:.2e} |
    | $K\to0$: $C_0/S_0$ | {_lim:.8f} |
    | Paridad: put (suma) − (call − $S_0$ + $Ke^{{-rT}}$) | {_parity:.2e} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 3 · ¿Cuánto aporta cada término de la suma?

    Cada barra es $P_k'\cdot\mathrm{BS}(r_k,\sigma_k)$. La línea muestra la suma acumulada: converge al
    precio de Merton (negro) y queda **por encima** de BS con $\sigma$ (rojo) cuando hay saltos negativos grandes.
    Cambia `K*` para ver cuándo la suma ya no se mueve: es la cota de Poisson en acción.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bk_K = mo.ui.slider(60, 140, step=1, value=100, label="K")
    bk_T = mo.ui.slider(0.1, 3.0, step=0.1, value=1.0, label="T")
    bk_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    bk_lam = mo.ui.slider(0.0, 8.0, step=0.1, value=1.0, label="λ")
    bk_muJ = mo.ui.slider(-0.5, 0.2, step=0.01, value=-0.20, label="μ_J")
    bk_sigJ = mo.ui.slider(0.01, 0.6, step=0.01, value=0.30, label="σ_J")
    bk_kmax = mo.ui.slider(2, 30, step=1, value=12, label="K* (términos)")
    mo.vstack(
        [
            mo.hstack([bk_K, bk_T, bk_kmax], justify="start"),
            mo.hstack([bk_sigma, bk_lam, bk_muJ, bk_sigJ], justify="start"),
        ]
    )
    return bk_K, bk_T, bk_kmax, bk_lam, bk_muJ, bk_sigJ, bk_sigma


@app.cell
def _(
    bk_K, bk_T, bk_kmax, bk_lam, bk_muJ, bk_sigJ, bk_sigma,
    bs_call, merton_breakdown, merton_call, mo, np, plt,
):
    _args = (100.0, bk_K.value, bk_T.value, 0.05, bk_sigma.value, bk_lam.value, bk_muJ.value, bk_sigJ.value)
    _k, _Pk, _rk, _sk, _bs = merton_breakdown(*_args, kmax=bk_kmax.value)
    _contrib = _Pk * _bs
    _cum = np.cumsum(_contrib)
    _C = merton_call(*_args)
    _bs0 = bs_call(100.0, bk_K.value, bk_T.value, 0.05, bk_sigma.value)

    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _ax.bar(_k, _contrib, color="C0", alpha=0.6, label=r"aporte $P_k'\,\mathrm{BS}_k$")
    _ax.plot(_k, _cum, "o-", color="C1", label="suma acumulada")
    _ax.axhline(_C, color="k", lw=1.4, label=f"Merton = {_C:.4f}")
    _ax.axhline(_bs0, color="r", ls="--", lw=1.2, label=f"BS(σ) = {_bs0:.4f}")
    _ax.set_xlabel("k")
    _ax.set_ylabel("precio de la call")
    _ax.legend(fontsize=8)
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {k} | {p:.4f} | {a:.4f} | {b:.4f} | {c:.4f} |"
        for k, p, a, b, c in zip(_k[:6], _Pk[:6], _rk[:6], _sk[:6], _contrib[:6])
    )
    _tbl = mo.md(
        f"""
    | k | $P_k'$ | $r_k$ | $\\sigma_k$ | aporte |
    |---|---|---|---|---|
    {_rows}

    Error de truncamiento con K* = {bk_kmax.value}: **{abs(_C - _cum[-1]):.2e}**
    """
    )
    mo.hstack([_fig, _tbl], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 4 · Sensibilidad del precio a un parámetro

    Elige **qué parámetro barrer**; los demás quedan fijos en los valores de los deslizadores.
    Compara con BS (mismo $\sigma$): la diferencia es lo que "cobra" el riesgo de salto.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sv_var = mo.ui.dropdown(
        {"λ": "lam", "μ_J": "muJ", "σ_J": "sigJ", "σ": "sigma", "K": "K", "T": "T"},
        value="λ",
        label="parámetro a barrer",
    )
    sv_K = mo.ui.slider(60, 140, step=1, value=100, label="K")
    sv_T = mo.ui.slider(0.1, 3.0, step=0.1, value=1.0, label="T")
    sv_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    sv_lam = mo.ui.slider(0.0, 5.0, step=0.1, value=1.0, label="λ")
    sv_muJ = mo.ui.slider(-0.5, 0.3, step=0.01, value=-0.20, label="μ_J")
    sv_sigJ = mo.ui.slider(0.01, 0.6, step=0.01, value=0.30, label="σ_J")
    mo.vstack(
        [
            sv_var,
            mo.hstack([sv_K, sv_T, sv_sigma], justify="start"),
            mo.hstack([sv_lam, sv_muJ, sv_sigJ], justify="start"),
        ]
    )
    return sv_K, sv_T, sv_lam, sv_muJ, sv_sigJ, sv_sigma, sv_var


@app.cell
def _(
    bs_call, merton_call, np, plt,
    sv_K, sv_T, sv_lam, sv_muJ, sv_sigJ, sv_sigma, sv_var,
):
    _base = dict(K=sv_K.value, T=sv_T.value, sigma=sv_sigma.value,
                 lam=sv_lam.value, muJ=sv_muJ.value, sigJ=sv_sigJ.value)
    _ranges = {"lam": (0.0, 5.0), "muJ": (-0.5, 0.3), "sigJ": (0.01, 0.6),
               "sigma": (0.05, 0.6), "K": (60.0, 140.0), "T": (0.1, 3.0)}
    _key = sv_var.value
    _grid = np.linspace(*_ranges[_key], 60)

    _C, _B = [], []
    for _v in _grid:
        _p = dict(_base)
        _p[_key] = _v
        _C.append(merton_call(100.0, _p["K"], _p["T"], 0.05, _p["sigma"], _p["lam"], _p["muJ"], _p["sigJ"]))
        _B.append(bs_call(100.0, _p["K"], _p["T"], 0.05, _p["sigma"]))
    _C, _B = np.array(_C), np.array(_B)

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.6))
    _axs[0].plot(_grid, _C, "k", lw=1.6, label="Merton")
    _axs[0].plot(_grid, _B, "r--", lw=1.2, label="BS (σ)")
    _axs[0].axvline(_base[_key], color="gray", lw=0.8)
    _axs[0].set_xlabel(_key)
    _axs[0].set_ylabel("call")
    _axs[0].legend()
    _axs[1].plot(_grid, _C - _B, "C0", lw=1.6)
    _axs[1].axhline(0, color="k", lw=0.8)
    _axs[1].axvline(_base[_key], color="gray", lw=0.8)
    _axs[1].set_xlabel(_key)
    _axs[1].set_title("Merton − BS (prima por saltos)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 5 · Mapa del precio en el plano $(\lambda,\mu_J)$

    ¿Dónde pesa más el riesgo de salto? Elige **precio** o **volatilidad implícita ATM**. Observa que
    con $\mu_J$ cercano a 0 y $\sigma_J$ pequeño, aumentar $\lambda$ casi no cambia nada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    hm_what = mo.ui.dropdown(["precio de la call", "IV ATM (%)"], value="precio de la call", label="mostrar")
    hm_T = mo.ui.slider(0.1, 3.0, step=0.1, value=1.0, label="T")
    hm_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    hm_sigJ = mo.ui.slider(0.01, 0.6, step=0.01, value=0.20, label="σ_J")
    mo.hstack([hm_what, hm_T, hm_sigma, hm_sigJ], justify="start")
    return hm_T, hm_sigJ, hm_sigma, hm_what


@app.cell
def _(hm_T, hm_sigJ, hm_sigma, hm_what, merton_call, merton_iv, np, plt):
    _lams = np.linspace(0.0, 5.0, 26)
    _mus = np.linspace(-0.5, 0.2, 26)

    def _f(lam, mu):
        if hm_what.value == "precio de la call":
            return merton_call(100.0, 100.0, hm_T.value, 0.05, hm_sigma.value, lam, mu, hm_sigJ.value)
        return 100 * merton_iv(100.0, 100.0, hm_T.value, 0.05, hm_sigma.value, lam, mu, hm_sigJ.value)

    _Z = np.array([[_f(l, m) for l in _lams] for m in _mus])
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    _pc = _ax.pcolormesh(_lams, _mus, _Z, shading="auto", cmap="viridis")
    _ax.contour(_lams, _mus, _Z, colors="w", linewidths=0.6)
    _fig.colorbar(_pc, ax=_ax, label=hm_what.value)
    _ax.set_xlabel("λ")
    _ax.set_ylabel("μ_J")
    _ax.grid(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Función característica

    Por independencia y condicionando en $N_T$ ($\mathbb E[e^{iuJ}]=e^{iu\mu_J-u^2\sigma_J^2/2}$):

    $$\phi_T(u)=\exp\!\Big[iu\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)T-\tfrac{\sigma^2u^2}{2}T+\lambda T\big(e^{iu\mu_J-u^2\sigma_J^2/2}-1\big)\Big]$$

    **Difusión gaussiana** + **deriva compensada** + **pgf de Poisson compuesta**.

    Es la entrada de los métodos de Fourier: con amortiguación $\alpha>0$, Carr–Madan expresa la call como transformada inversa de $\phi_T$ evaluada en $u-i(\alpha+1)$.

    > Hoy calibramos por MLE sobre retornos; la calibración sobre superficies (vía Fourier) es la **tarea de extensión**.

    **Verificación numérica.** La tabla compara la suma de Merton con la inversión de Fourier de Carr–Madan (parámetros de la Tarea 2).
    """)
    return


@app.cell
def _(comp_m, math, np, simpson):
    def merton_cf(u, T, r, sigma, lam, muJ, sigJ):
        """Función característica de ln(S_T/S_0). Admite u complejo."""
        m = comp_m(muJ, sigJ)
        return np.exp(
            1j * u * (r - lam * m - 0.5 * sigma**2) * T
            - 0.5 * sigma**2 * u**2 * T
            + lam * T * (np.exp(1j * u * muJ - 0.5 * u**2 * sigJ**2) - 1)
        )

    def carr_madan_call(S0, K, T, r, sigma, lam, muJ, sigJ, alpha=1.5, vmax=200.0, npts=40001):
        v = np.linspace(0.0, vmax, npts)
        u = v - (alpha + 1) * 1j
        phi = np.exp(1j * u * math.log(S0)) * merton_cf(u, T, r, sigma, lam, muJ, sigJ)
        psi = math.exp(-r * T) * phi / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
        k = math.log(K)
        integrand = np.real(np.exp(-1j * v * k) * psi)
        return math.exp(-alpha * k) / math.pi * simpson(integrand, x=v)

    return carr_madan_call, merton_cf


@app.cell(hide_code=True)
def _(mo):
    f_alpha = mo.ui.slider(0.5, 3.0, step=0.25, value=1.5, label="α (amortiguación)")
    f_T = mo.ui.slider(0.1, 3.0, step=0.1, value=1.0, label="T")
    mo.hstack([f_alpha, f_T], justify="start")
    return f_T, f_alpha


@app.cell
def _(carr_madan_call, f_T, f_alpha, merton_call, mo):
    _par = dict(S0=100.0, T=f_T.value, r=0.05, sigma=0.2, lam=1.0, muJ=-0.2, sigJ=0.3)
    _rows = []
    for _K in [70, 85, 100, 115, 130]:
        _s = merton_call(K=float(_K), **_par)
        _f = carr_madan_call(K=float(_K), alpha=f_alpha.value, **_par)
        _rows.append(f"| {_K} | {_s:.6f} | {_f:.6f} | {abs(_s - _f):.1e} |")
    mo.md("| K | suma de Merton | Carr–Madan | diferencia |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Smile de volatilidad: qué predice el modelo

    - Mezcla Poisson de normales: curtosis en exceso positiva y asimetría con el signo de $\mu_J$ ⇒ **smile/skew**.
    - **Estructura temporal.** $\kappa_n\propto T$ ⇒ $\gamma_1=O(T^{-1/2})$, $\gamma_2=O(T^{-1})$: el smile **se aplana** al crecer $T$. IV ATM a plazo largo $\to\sqrt{\sigma^2+\lambda(\sigma_J^2+\mu_J^2)}$ (heurística TCL).
    - **Plazo corto.** Precio OTM $\approx\lambda T\,\mathbb E[(S_0Y-K)^+]$ (lineal en $T$): la IV de strikes OTM crece sin cota cuando $T\to0$.
    - **Datos.** SPX y SX5E: skew negativo persistente, coherente con $\mu_J<0$.

    > **Límite del modelo:** el skew empírico a plazos largos decae más despacio que $T^{-1/2}$; con saltos i.i.d. el MJD no lo reproduce.

    *(El smile calculado con el modelo está en la Hora 2, sección "De la calibración a la decisión".)*

    ---
    # Limitaciones estructurales

    1. **Mercado incompleto.** Dos fuentes de riesgo ($W$, $N$) y un solo activo: no hay cartera replicante; hay infinitas medidas martingala equivalentes. Se necesita un criterio para elegir $\mathbb Q$ (varianza mínima, entropía mínima, Esscher, utilidad exponencial).
    2. **Actividad finita.** $\nu(\mathbb R)=\lambda<\infty$: no captura actividad infinita (VG, CGMY, NIG).
    3. **Saltos lognormales i.i.d.** Asimetría con $\mu_J<0$, pero no la estructura temporal del skew.
    4. **Identificación.** Con $\lambda\Delta$ pequeño (datos diarios) los saltos raros se confunden con ruido difusivo: **identificación débil** (se cuantifica en la Hora 2).

    ---
    # Extensiones y transición

    **Para conocer, no desarrollar hoy:**

    - **Kou (2002):** saltos exponenciales dobles ⇒ fórmula cerrada, mejor skew en acciones.
    - **Bates (1996):** Heston + saltos ⇒ estándar de industria en índices.
    - **Duffie–Pan–Singleton (2000):** marco afín con función característica cerrada.

    > **Transición.** El modelo es incompleto y sus parámetros no son observables. ¿Cómo lo simulamos *exactamente*, cómo verificamos el precio y cómo lo calibramos con datos de acciones e índices?
    """)
    return


# ======================================================================
#  HORA 2 · SIMULACIÓN
# ======================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    **HORA 2**

    # Simulación y aplicación

    ---
    # Simulación exacta de la ley terminal

    - **Exacta, no Euler:** incrementos de $\ln S$ independientes y estacionarios con ley conocida; dado $N_\Delta=k$, la suma de saltos es $\mathcal N(k\mu_J,k\sigma_J^2)$. Sin error de discretización en la **ley** de $S_{t_j}$.
    - **Si $dN=2$:** la suma es $\mathcal N(2\mu_J,2\sigma_J^2)$, **no** $2\cdot\mathcal N(\mu_J,\sigma_J^2)$.

    Abajo, el código: `simulate_merton` simula **trayectorias completas** (rejilla de $n$ pasos) y
    `simulate_terminal` simula solo $S_T$ **en un paso** (basta para precios europeos).
    El argumento `compensate=False` permite quitar $-\lambda m$ para romper la martingala a propósito.
    """)
    return


@app.cell
def _(comp_m, math, np):
    def simulate_merton(S0, r, sigma, lam, muJ, sigmaJ, T, n, n_paths, seed=0, compensate=True):
        """Simulación exacta por incrementos de ln S (código de la diapositiva)."""
        rng = np.random.default_rng(seed)
        dt = T / n
        m = np.exp(muJ + 0.5 * sigmaJ**2) - 1 if compensate else 0.0
        dN = rng.poisson(lam * dt, size=(n_paths, n))
        J = rng.normal(muJ * dN, sigmaJ * np.sqrt(np.maximum(dN, 1))) * (dN > 0)
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n))
        log_inc = (r - lam * m - 0.5 * sigma**2) * dt + sigma * dW + J
        log_S = np.log(S0) + np.cumsum(log_inc, axis=1)
        return np.concatenate([np.full((n_paths, 1), S0), np.exp(log_S)], axis=1), dN

    def simulate_terminal(S0, r, sigma, lam, muJ, sigJ, T, n_paths, seed=0, compensate=True):
        """Ley terminal exacta en un paso: N_T ~ Poisson, suma de saltos normal."""
        rng = np.random.default_rng(seed)
        m = comp_m(muJ, sigJ) if compensate else 0.0
        N = rng.poisson(lam * T, size=n_paths)
        jumps = rng.normal(muJ * N, sigJ * np.sqrt(N))
        Z = rng.standard_normal(n_paths)
        x = (r - lam * m - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z + jumps
        return S0 * np.exp(x), N, x

    return simulate_merton, simulate_terminal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Resultado 1 · Trayectorias de una acción

    Por defecto: $S_0=100$, $r=0.05$, $\sigma=0.2$, $\lambda=5$, $\mu_J=-0.1$, $\sigma_J=0.2$, $T=1$ (acción con resultados trimestrales).
    En la diapositiva, con $2\times10^6$ trayectorias: $\mathbb E[S_T]=105.090\pm0.039$ (teoría 105.127), $\mathbb E[N_T]=5.0007$, $\mathrm{Var}(N_T)=4.9975$, $z=-0.94$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    s1_S0 = mo.ui.number(value=100.0, start=1, stop=1000, step=1, label="S0")
    s1_r = mo.ui.slider(0.0, 0.15, step=0.005, value=0.05, label="r")
    s1_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    s1_lam = mo.ui.slider(0.0, 20.0, step=0.5, value=5.0, label="λ")
    s1_muJ = mo.ui.slider(-0.4, 0.2, step=0.01, value=-0.10, label="μ_J")
    s1_sigJ = mo.ui.slider(0.01, 0.5, step=0.01, value=0.20, label="σ_J")
    s1_T = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="T")
    s1_npaths = mo.ui.dropdown(
        {"2 · 10⁵": 200_000, "10⁶": 1_000_000, "2 · 10⁶": 2_000_000},
        value="10⁶",
        label="Trayectorias (chequeos)",
    )
    s1_seed = mo.ui.number(value=0, start=0, stop=10_000, step=1, label="semilla")
    s1_nocomp = mo.ui.checkbox(value=False, label="Omitir compensador −λm")
    mo.vstack(
        [
            mo.hstack([s1_S0, s1_r, s1_sigma], justify="start"),
            mo.hstack([s1_lam, s1_muJ, s1_sigJ], justify="start"),
            mo.hstack([s1_T, s1_npaths, s1_seed, s1_nocomp], justify="start"),
        ]
    )
    return (
        s1_S0,
        s1_T,
        s1_lam,
        s1_muJ,
        s1_nocomp,
        s1_npaths,
        s1_r,
        s1_seed,
        s1_sigJ,
        s1_sigma,
    )


@app.cell
def _(
    comp_m,
    mo,
    np,
    plt,
    s1_S0,
    s1_T,
    s1_lam,
    s1_muJ,
    s1_nocomp,
    s1_npaths,
    s1_r,
    s1_seed,
    s1_sigJ,
    s1_sigma,
    simulate_merton,
    simulate_terminal,
):
    _S0, _r, _sg = s1_S0.value, s1_r.value, s1_sigma.value
    _lam, _mu, _sj, _T = s1_lam.value, s1_muJ.value, s1_sigJ.value, s1_T.value
    _comp = not s1_nocomp.value

    _n = max(int(252 * _T), 20)
    _paths, _dN = simulate_merton(
        _S0, _r, _sg, _lam, _mu, _sj, _T, _n, 12, seed=s1_seed.value, compensate=_comp
    )
    _t = np.linspace(0, _T, _n + 1)

    _fig, _ax = plt.subplots()
    for _i in range(_paths.shape[0]):
        _ax.plot(_t, _paths[_i], lw=0.9)
        _jump_idx = np.where(_dN[_i] > 0)[0] + 1
        _ax.scatter(_t[_jump_idx], _paths[_i, _jump_idx], s=10, c="k", zorder=3)
    _ax.plot(_t, _S0 * np.exp(_r * _t), "k--", lw=1.5, label=r"$S_0e^{rt}$")
    _ax.set_xlabel("t")
    _ax.set_ylabel("S")
    _ax.set_title("Trayectorias (puntos negros = días con salto)")
    _ax.legend()
    _fig.tight_layout()

    _ST, _N, _ = simulate_terminal(
        _S0, _r, _sg, _lam, _mu, _sj, _T, s1_npaths.value, seed=s1_seed.value + 1, compensate=_comp
    )
    _mean, _se = _ST.mean(), _ST.std(ddof=1) / np.sqrt(len(_ST))
    _theory = _S0 * np.exp(_r * _T)
    _z = (_mean - _theory) / _se
    _m = comp_m(_mu, _sj)
    _extra = ""
    if not _comp:
        _extra = f"| teoría sin compensador $S_0e^{{(r+\\lambda m)T}}$ | | {_S0 * np.exp((_r + _lam * _m) * _T):.3f} |\n"

    _tbl = mo.md(
        rf"""
    | | simulado | teoría |
    |---|---|---|
    | $\mathbb E[S_T]$ | {_mean:.3f} ± {_se:.3f} | {_theory:.3f} |
    | $\mathbb E[N_T]$ | {_N.mean():.4f} | {_lam * _T:.4f} |
    | $\mathrm{{Var}}(N_T)$ | {_N.var():.4f} | {_lam * _T:.4f} |
    {_extra}
    $z = {_z:+.2f}$ → {"la martingala **no se rechaza**" if abs(_z) < 1.96 else "**la martingala se rechaza**" + (" (era lo esperado: sin compensador)" if not _comp else " (revisar el compensador)")}
    """
    )
    mo.hstack([_fig, _tbl], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 6 · Martingala y conteo de saltos, lado a lado

    Usa los **mismos parámetros y la misma semilla** del Resultado 1. Izquierda: media móvil de $S_T$
    con compensador (negro) y **sin** compensador (rojo); las líneas punteadas son las teorías
    $S_0e^{rT}$ y $S_0e^{(r+\lambda m)T}$. Derecha: histograma de $N_T$ contra la pmf de Poisson.
    """)
    return


@app.cell
def _(
    comp_m, np, plt, poisson,
    s1_S0, s1_T, s1_lam, s1_muJ, s1_npaths, s1_r, s1_seed, s1_sigJ, s1_sigma,
    simulate_terminal,
):
    _p = (s1_S0.value, s1_r.value, s1_sigma.value, s1_lam.value, s1_muJ.value, s1_sigJ.value, s1_T.value)
    _n = s1_npaths.value
    _ST_c, _N, _ = simulate_terminal(*_p, _n, seed=s1_seed.value + 1, compensate=True)
    _ST_n, _, _ = simulate_terminal(*_p, _n, seed=s1_seed.value + 1, compensate=False)

    _sizes = np.unique(np.logspace(2, np.log10(_n), 60).astype(int))
    _run_c = np.cumsum(_ST_c)[_sizes - 1] / _sizes
    _run_n = np.cumsum(_ST_n)[_sizes - 1] / _sizes
    _se = _ST_c.std(ddof=1) / np.sqrt(_sizes)
    _th = _p[0] * np.exp(_p[1] * _p[6])
    _th_n = _p[0] * np.exp((_p[1] + _p[3] * comp_m(_p[4], _p[5])) * _p[6])

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8))
    _axs[0].plot(_sizes, _run_c, "k", label="con compensador")
    _axs[0].fill_between(_sizes, _run_c - 1.96 * _se, _run_c + 1.96 * _se, color="gray", alpha=0.25)
    _axs[0].plot(_sizes, _run_n, "r", label="sin compensador")
    _axs[0].axhline(_th, color="k", ls=":", label=r"$S_0e^{rT}$")
    _axs[0].axhline(_th_n, color="r", ls=":", label=r"$S_0e^{(r+\lambda m)T}$")
    _axs[0].set_xscale("log")
    _axs[0].set_xlabel("número de trayectorias")
    _axs[0].set_title(r"Media móvil de $S_T$")
    _axs[0].legend(fontsize=8)

    _kk = np.arange(0, int(_N.max()) + 1)
    _axs[1].hist(_N, bins=np.arange(-0.5, _N.max() + 1.5), density=True, alpha=0.5, label="simulado")
    _axs[1].plot(_kk, poisson.pmf(_kk, _p[3] * _p[6]), "ko-", ms=4, label="Poisson(λT)")
    _axs[1].set_xlabel(r"$N_T$")
    _axs[1].set_title("Número de saltos")
    _axs[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Resultado 2 · Momentos: teoría vs. simulación

    Mismos parámetros, $T=1$: $\kappa_2=0.29$. En la diapositiva: $\gamma_1=-0.4162$ (sim. $-0.4152$), $\gamma_2=0.4340$ (sim. $0.4290$).

    La normal con igual media y varianza (BS) no reproduce la cola izquierda: la mezcla pone masa en $k\ge1$ saltos negativos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    s2_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    s2_lam = mo.ui.slider(0.0, 20.0, step=0.5, value=5.0, label="λ")
    s2_muJ = mo.ui.slider(-0.4, 0.2, step=0.01, value=-0.10, label="μ_J")
    s2_sigJ = mo.ui.slider(0.01, 0.5, step=0.01, value=0.20, label="σ_J")
    s2_T = mo.ui.slider(0.05, 5.0, step=0.05, value=1.0, label="T")
    mo.vstack(
        [
            mo.hstack([s2_sigma, s2_lam, s2_T], justify="start"),
            mo.hstack([s2_muJ, s2_sigJ], justify="start"),
        ]
    )
    return s2_T, s2_lam, s2_muJ, s2_sigJ, s2_sigma


@app.cell
def _(
    comp_m,
    mo,
    norm,
    np,
    plt,
    poisson,
    s2_T,
    s2_lam,
    s2_muJ,
    s2_sigJ,
    s2_sigma,
    simulate_terminal,
):
    _r = 0.05
    _sg, _lam, _mu, _sj, _T = (
        s2_sigma.value,
        s2_lam.value,
        s2_muJ.value,
        s2_sigJ.value,
        s2_T.value,
    )
    _m = comp_m(_mu, _sj)

    _k2 = _sg**2 * _T + _lam * _T * (_sj**2 + _mu**2)
    _k3 = _lam * _T * (_mu**3 + 3 * _mu * _sj**2)
    _k4 = _lam * _T * (_mu**4 + 6 * _mu**2 * _sj**2 + 3 * _sj**4)
    _g1_th, _g2_th = _k3 / _k2**1.5, _k4 / _k2**2

    _, _, _x = simulate_terminal(100.0, _r, _sg, _lam, _mu, _sj, _T, 1_000_000, seed=7)
    _xc = _x - _x.mean()
    _g1_sim = (_xc**3).mean() / _xc.var() ** 1.5
    _g2_sim = (_xc**4).mean() / _xc.var() ** 2 - 3

    _grid = np.linspace(_x.mean() - 5 * np.sqrt(_k2), _x.mean() + 5 * np.sqrt(_k2), 600)
    _kk = np.arange(0, 80)
    _w = poisson.pmf(_kk, _lam * _T)
    _mu_k = (_r - _lam * _m - 0.5 * _sg**2) * _T + _kk * _mu
    _s_k = np.sqrt(_sg**2 * _T + _kk * _sj**2)
    _dens = (_w[None, :] * norm.pdf(_grid[:, None], _mu_k[None, :], _s_k[None, :])).sum(1)
    _normal = norm.pdf(_grid, _x.mean(), np.sqrt(_k2))

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 4))
    for _a, _logy in zip(_axs, [False, True]):
        _a.hist(_x, bins=200, density=True, alpha=0.35, color="gray", label="simulado")
        _a.plot(_grid, _dens, "k", lw=1.5, label="mezcla Poisson")
        _a.plot(_grid, _normal, "r--", lw=1.2, label="normal (BS)")
        _a.set_xlabel(r"$\ln(S_T/S_0)$")
        if _logy:
            _a.set_yscale("log")
            _a.set_ylim(1e-4, None)
            _a.set_title("escala log (colas)")
        else:
            _a.set_title("densidad")
    _axs[0].legend()
    _fig.tight_layout()

    _tbl = mo.md(
        rf"""
    | | teoría | simulado |
    |---|---|---|
    | $\kappa_2$ | {_k2:.4f} | {_x.var():.4f} |
    | $\gamma_1$ | {_g1_th:+.4f} | {_g1_sim:+.4f} |
    | $\gamma_2$ | {_g2_th:+.4f} | {_g2_sim:+.4f} |
    """
    )
    mo.vstack([_fig, _tbl])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 7 · ¿Cuántas veces más probable es una caída extrema?

    Misma configuración del Resultado 2. Estandarizamos $\ln(S_T/S_0)$ y contamos la frecuencia de
    caídas peores que $-k$ desviaciones. La razón **simulado / normal** es el "error de BS" en las colas.
    A la derecha, el QQ-plot: una normal sería la diagonal.
    """)
    return


@app.cell
def _(mo, norm, np, plt, s2_T, s2_lam, s2_muJ, s2_sigJ, s2_sigma, simulate_terminal):
    _, _, _x = simulate_terminal(
        100.0, 0.05, s2_sigma.value, s2_lam.value, s2_muJ.value, s2_sigJ.value,
        s2_T.value, 1_000_000, seed=3,
    )
    _z = (_x - _x.mean()) / _x.std()

    _ks = np.array([2, 3, 4, 5])
    _emp = np.array([(_z < -k).mean() for k in _ks])
    _nor = norm.cdf(-_ks)
    _rows = "\n".join(
        f"| −{k}σ | {e:.2e} | {n:.2e} | {(e / n):.1f}× |" for k, e, n in zip(_ks, _emp, _nor)
    )

    _probs = np.linspace(0.0005, 0.9995, 400)
    _fig, _ax = plt.subplots(figsize=(4.8, 4.2))
    _ax.plot(norm.ppf(_probs), np.quantile(_z, _probs), "C0", lw=1.6)
    _ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", lw=1)
    _ax.set_xlabel("cuantiles normal")
    _ax.set_ylabel("cuantiles simulados")
    _ax.set_title("QQ-plot")
    _fig.tight_layout()

    _tbl = mo.md(
        f"""
    | caída | frecuencia simulada | normal | razón |
    |---|---|---|---|
    {_rows}
    """
    )
    mo.hstack([_tbl, _fig], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Suma truncada de Merton

    En el cuaderno la suma está **vectorizada** (`merton_terms`, definida en la Hora 1); la versión con
    bucle de la diapositiva hace exactamente lo mismo, término a término:

    ```python
    def merton_call_loop(S0, K, T, r, sigma, lam, muJ, sigmaJ, Ktrunc=30):
        m = math.exp(muJ + 0.5 * sigmaJ**2) - 1
        price = 0.0
        for k in range(Ktrunc + 1):
            Pk = math.exp(-lam * T) * (lam * T)**k / math.factorial(k)
            rk = r - lam * m + k * (muJ + 0.5 * sigmaJ**2) / T
            sk = math.sqrt(sigma**2 + k * sigmaJ**2 / T)
            d1 = (math.log(S0 / K) + (rk + 0.5 * sk**2) * T) / (sk * math.sqrt(T))
            d2 = d1 - sk * math.sqrt(T)
            price += Pk * (S0 * math.exp((rk - r) * T) * norm.cdf(d1)
                           - K * math.exp(-r * T) * norm.cdf(d2))
        return price
    ```

    > **Tarea 2 (8 min, parejas).** $S_0=K=100$, $r=0.05$, $\sigma=0.2$, $T=1$, $\lambda=1$, $\mu_J=-0.2$, $\sigma_J=0.3$. Compara con Monte Carlo, varía $K^*\in\{5,10,15,20,30\}$ y justifica $K^*$ con la cota de Poisson.

    ---
    # Resultado 3 · Suma vs. Monte Carlo

    Los parámetros por defecto son los de la Tarea 2. En la diapositiva: Merton ($K^*=60$) $=16.8617$; Monte Carlo ($2\times10^6$) $=16.8440\pm0.0180$; BS ($\sigma=0.2$) $=10.4506$; put por paridad $=11.9846$; $z=-0.98$.

    Los saltos negativos grandes **elevan el precio** frente a BS: $\kappa_2=0.04+1\cdot(0.09+0.04)=0.17$ vs. $0.04$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    s3_S0 = mo.ui.number(value=100.0, start=1, stop=1000, step=1, label="S0")
    s3_K = mo.ui.number(value=100.0, start=1, stop=1000, step=1, label="K")
    s3_T = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="T")
    s3_r = mo.ui.slider(0.0, 0.15, step=0.005, value=0.05, label="r")
    s3_sigma = mo.ui.slider(0.05, 0.6, step=0.01, value=0.20, label="σ")
    s3_lam = mo.ui.slider(0.0, 10.0, step=0.1, value=1.0, label="λ")
    s3_muJ = mo.ui.slider(-0.5, 0.2, step=0.01, value=-0.20, label="μ_J")
    s3_sigJ = mo.ui.slider(0.01, 0.6, step=0.01, value=0.30, label="σ_J")
    s3_Kstar = mo.ui.slider(0, 60, step=1, value=30, label="K* (truncamiento)")
    s3_mc = mo.ui.dropdown(
        {"2 · 10⁵": 200_000, "10⁶": 1_000_000, "2 · 10⁶": 2_000_000},
        value="10⁶",
        label="Monte Carlo",
    )
    mo.vstack(
        [
            mo.hstack([s3_S0, s3_K, s3_T, s3_r], justify="start"),
            mo.hstack([s3_sigma, s3_lam, s3_muJ, s3_sigJ], justify="start"),
            mo.hstack([s3_Kstar, s3_mc], justify="start"),
        ]
    )
    return (
        s3_K,
        s3_Kstar,
        s3_S0,
        s3_T,
        s3_lam,
        s3_mc,
        s3_muJ,
        s3_r,
        s3_sigJ,
        s3_sigma,
    )


@app.cell
def _(
    bs_call,
    bs_put,
    math,
    merton_call,
    mo,
    np,
    plt,
    s3_K,
    s3_Kstar,
    s3_S0,
    s3_T,
    s3_lam,
    s3_mc,
    s3_muJ,
    s3_r,
    s3_sigJ,
    s3_sigma,
    simulate_terminal,
):
    _S0, _K, _T, _r = s3_S0.value, s3_K.value, s3_T.value, s3_r.value
    _sg, _lam, _mu, _sj = s3_sigma.value, s3_lam.value, s3_muJ.value, s3_sigJ.value

    _C = merton_call(_S0, _K, _T, _r, _sg, _lam, _mu, _sj, kmax=s3_Kstar.value)
    _C_ref = merton_call(_S0, _K, _T, _r, _sg, _lam, _mu, _sj)  # truncamiento automático
    _put = _C_ref - _S0 + _K * math.exp(-_r * _T)
    _bs = bs_call(_S0, _K, _T, _r, _sg)

    _NMC = s3_mc.value
    _ST, _, _ = simulate_terminal(_S0, _r, _sg, _lam, _mu, _sj, _T, _NMC, seed=11)
    _pay = np.exp(-_r * _T) * np.maximum(_ST - _K, 0.0)
    _mc, _se = _pay.mean(), _pay.std(ddof=1) / math.sqrt(_NMC)
    _z = (_C_ref - _mc) / _se

    _sizes = np.unique(np.logspace(3, math.log10(_NMC), 40).astype(int))
    _cum = np.cumsum(_pay)
    _run = _cum[_sizes - 1] / _sizes
    _run_se = np.array([_pay[:n].std(ddof=1) / math.sqrt(n) for n in _sizes])

    _fig, _ax = plt.subplots()
    _ax.plot(_sizes, _run, "k", lw=1.2, label="Monte Carlo")
    _ax.fill_between(_sizes, _run - 1.96 * _run_se, _run + 1.96 * _run_se, color="gray", alpha=0.3)
    _ax.axhline(_C_ref, color="C0", lw=1.5, label="Merton (suma)")
    _ax.axhline(_bs, color="C3", ls="--", lw=1.2, label="Black–Scholes (σ)")
    _ax.set_xscale("log")
    _ax.set_xlabel("número de trayectorias")
    _ax.set_ylabel("precio de la call")
    _ax.legend()
    _fig.tight_layout()

    _tbl = mo.md(
        rf"""
    | Método | Call |
    |---|---|
    | **Merton (K* = {s3_Kstar.value})** | **{_C:.4f}** |
    | Merton (truncamiento automático) | {_C_ref:.4f} |
    | Monte Carlo ({_NMC:.0e}) | {_mc:.4f} ± {_se:.4f} |
    | Black–Scholes (σ = {_sg}) | {_bs:.4f} |
    | Put (paridad) | {_put:.4f} |
    | Put BS (σ) | {bs_put(_S0, _K, _T, _r, _sg):.4f} |

    $z={_z:+.2f}$ (suma vs. MC)
    """
    )
    mo.hstack([_fig, _tbl], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Resultado 4 · Error de truncamiento y cota

    $\lambda'=\lambda(1+m)=0.856$ con los parámetros de la Tarea 2; cota $=S_0\,\mathbb P(N'>K^*)$.

    La cota **siempre** domina; bajo $10^{-14}$ el error observado es ruido de punto flotante.
    Los parámetros son los de la sección anterior.
    """)
    return


@app.cell
def _(
    comp_m,
    math,
    merton_call,
    mo,
    np,
    plt,
    poisson,
    s3_K,
    s3_S0,
    s3_T,
    s3_lam,
    s3_muJ,
    s3_r,
    s3_sigJ,
    s3_sigma,
):
    _S0, _K, _T, _r = s3_S0.value, s3_K.value, s3_T.value, s3_r.value
    _sg, _lam, _mu, _sj = s3_sigma.value, s3_lam.value, s3_muJ.value, s3_sigJ.value
    _lamp = max(_lam * (1 + comp_m(_mu, _sj)), 1e-12)
    _ref = merton_call(_S0, _K, _T, _r, _sg, _lam, _mu, _sj)

    _ks = np.arange(1, 31)
    _err = np.array(
        [abs(_ref - merton_call(_S0, _K, _T, _r, _sg, _lam, _mu, _sj, kmax=int(k))) for k in _ks]
    )
    _bound = _S0 * poisson.sf(_ks, _lamp * _T)
    _chern = np.array(
        [
            _S0 * math.exp(-_lamp * _T + k * (1 + math.log(_lamp * _T / k))) if k > _lamp * _T else np.nan
            for k in _ks
        ]
    )

    _fig, _ax = plt.subplots()
    _ax.semilogy(_ks, np.maximum(_err, 1e-17), "o-", label="error real")
    _ax.semilogy(_ks, np.maximum(_bound, 1e-40), "s--", label=r"cota $S_0\,P(N'>K^*)$")
    _ax.semilogy(_ks, _chern, "^:", label="cota de Chernoff")
    _ax.axhline(1e-15, color="gray", lw=0.8)
    _ax.set_xlabel("K*")
    _ax.set_ylabel("error absoluto")
    _ax.set_ylim(1e-18, 1e2)
    _ax.legend()
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {k} | {max(_err[k - 1], 0):.1e} | {_bound[k - 1]:.1e} |" for k in [5, 10, 15, 20, 30]
    )
    _tbl = mo.md(
        rf"""
    $\lambda'={_lamp:.3f}$

    | K* | error real | cota |
    |---|---|---|
    {_rows}
    """
    )
    mo.hstack([_fig, _tbl], widths=[3, 2], align="center")
    return


# ----------------------------------------------------------------------
# Calibración
# ----------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Calibración por máxima verosimilitud

    *Calibrar es el problema inverso:* dados $x_1,\dots,x_n$ con paso $\Delta$, hallar $\theta=(\sigma,\lambda,\mu_J,\sigma_J)$.

    $$\ell(\theta)=\sum_{i=1}^n\ln\!\Big[\sum_{k\ge0}\frac{e^{-\lambda\Delta}(\lambda\Delta)^k}{k!}\,\varphi\big(x_i;\mu_k^{(\Delta)},s_k^{2,(\Delta)}\big)\Big]$$

    $$\mu_k^{(\Delta)}=\big(\alpha-\lambda m-\tfrac{\sigma^2}{2}\big)\Delta+k\mu_J,\qquad s_k^{2,(\Delta)}=\sigma^2\Delta+k\sigma_J^2$$

    - **Medida física.** Los datos viven bajo $\mathbb P$: deriva $\alpha$ (aquí $\alpha=r$ conocida, por simplicidad); en datos reales $\alpha$ es un parámetro más, muy débilmente identificado.
    - **Degeneración.** Con varianza libre la verosimilitud no está acotada si $\sigma\to0$: se impone $\sigma\ge\sigma_{\min}>0$ (L-BFGS-B con cotas).
    - **Numérica.** `logsumexp` sobre $k=0,\dots,K$; con $\lambda\Delta\approx0.02$, $K=20$ basta.
    - **Modelo incorrecto:** la MLE converge al miembro de la familia más cercano en divergencia **KL**.

    **Del papel al código.** `nll` es $-\ell(\theta)$: una matriz $n\times(K+1)$ de log-densidades
    normales más los log-pesos de Poisson, comprimida con `logsumexp`. `fit` la minimiza;
    `run_mle` lo repite desde varios puntos iniciales y calcula errores estándar con la Hessiana numérica.
    """)
    return


@app.cell
def _(comp_m, gammaln, logsumexp, minimize, np):
    DT = 1 / 252
    ALPHA = 0.02
    BOUNDS = [(0.02, 1.0), (0.05, 60.0), (-0.5, 0.5), (0.005, 0.5)]
    NAMES = ["σ", "λ", "μ_J", "σ_J"]

    def nll(theta, x, dt=DT, alpha=ALPHA, K=20):
        """Menos log-verosimilitud de retornos x con paso dt (mezcla Poisson de normales)."""
        sigma, lam, muJ, sigJ = theta
        m = comp_m(muJ, sigJ)
        k = np.arange(K + 1)
        logw = -lam * dt + k * np.log(lam * dt) - gammaln(k + 1)          # log P_k
        mu_k = (alpha - lam * m - 0.5 * sigma**2) * dt + k * muJ
        s2_k = sigma**2 * dt + k * sigJ**2
        logphi = -0.5 * np.log(2 * np.pi * s2_k) - 0.5 * (x[:, None] - mu_k) ** 2 / s2_k
        return -logsumexp(logw + logphi, axis=1).sum()

    def fit(x, theta_init, alpha=ALPHA, fixed_lam=None):
        """MLE con L-BFGS-B; si se fija λ devuelve la verosimilitud perfilada."""
        if fixed_lam is None:
            return minimize(nll, theta_init, args=(x, DT, alpha), method="L-BFGS-B", bounds=BOUNDS)

        def f(p):
            return nll([p[0], fixed_lam, p[1], p[2]], x, DT, alpha)

        p0 = [theta_init[0], theta_init[2], theta_init[3]]
        b = [BOUNDS[0], BOUNDS[2], BOUNDS[3]]
        return minimize(f, p0, method="L-BFGS-B", bounds=b)

    def num_hessian(f, theta, rel=1e-3):
        theta = np.asarray(theta, float)
        p = len(theta)
        h = rel * np.maximum(np.abs(theta), 1e-2)
        H = np.zeros((p, p))
        f0 = f(theta)
        for i in range(p):
            for j in range(i, p):
                ei, ej = np.zeros(p), np.zeros(p)
                ei[i], ej[j] = h[i], h[j]
                if i == j:
                    H[i, i] = (f(theta + ei) - 2 * f0 + f(theta - ei)) / h[i] ** 2
                else:
                    H[i, j] = H[j, i] = (
                        f(theta + ei + ej) - f(theta + ei - ej) - f(theta - ei + ej) + f(theta - ei - ej)
                    ) / (4 * h[i] * h[j])
        return H

    def run_mle(x, starts, alpha=ALPHA):
        """Multi-inicio + errores estándar (Hessiana numérica) + número de condición."""
        fits = [fit(x, s, alpha=alpha) for s in starts]
        best = min(fits, key=lambda r: r.fun)
        H = num_hessian(lambda th: nll(th, x, DT, alpha), best.x)
        try:
            se = np.sqrt(np.abs(np.diag(np.linalg.inv(H))))
        except np.linalg.LinAlgError:
            se = np.full(4, np.nan)
        return best, fits, se, np.linalg.cond(H)

    def simulate_returns(n, theta, seed, alpha=ALPHA):
        """n retornos diarios simulados bajo θ = (σ, λ, μ_J, σ_J)."""
        sigma, lam, muJ, sigJ = theta
        rng = np.random.default_rng(seed)
        m = comp_m(muJ, sigJ)
        N = rng.poisson(lam * DT, n)
        J = rng.normal(muJ * N, sigJ * np.sqrt(N))
        return (alpha - lam * m - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * rng.standard_normal(n) + J

    return (
        ALPHA,
        BOUNDS,
        DT,
        NAMES,
        fit,
        nll,
        num_hessian,
        run_mle,
        simulate_returns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Resultado 5 · Calibración sobre datos sintéticos

    6048 retornos diarios (24 años, $\Delta=1/252$) con $\theta_0=(0.18,\,5,\,-0.03,\,0.04)$, $\alpha=r=0.02$, 5 puntos iniciales. Permite **conocer la verdad**.

    En la diapositiva: $\hat\lambda=4.147\pm0.687$; $\hat\sigma=0.1814$; $\hat\mu_J=-0.0412$; $\hat\sigma_J=0.0405$; muestra con asimetría $-1.69$ y exceso de curtosis $12.83$; número de condición de la Hessiana $\approx1.6\times10^5$.
    Otra semilla da otra muestra: el patrón (σ y σ_J bien identificados, λ con error relativo de 15–40 %) es lo que se repite.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    s5_years = mo.ui.slider(2, 40, step=1, value=24, label="años de datos")
    s5_sigma = mo.ui.slider(0.05, 0.4, step=0.01, value=0.18, label="σ verdadero")
    s5_lam = mo.ui.slider(0.5, 20.0, step=0.5, value=5.0, label="λ verdadero")
    s5_muJ = mo.ui.slider(-0.15, 0.05, step=0.005, value=-0.03, label="μ_J verdadero")
    s5_sigJ = mo.ui.slider(0.01, 0.15, step=0.005, value=0.04, label="σ_J verdadero")
    s5_seed = mo.ui.number(value=3, start=0, stop=10_000, step=1, label="semilla")
    s5_run = mo.ui.run_button(label="Estimar")
    mo.vstack(
        [
            mo.hstack([s5_years, s5_sigma, s5_lam], justify="start"),
            mo.hstack([s5_muJ, s5_sigJ, s5_seed], justify="start"),
            s5_run,
        ]
    )
    return s5_lam, s5_muJ, s5_run, s5_seed, s5_sigJ, s5_sigma, s5_years


@app.cell
def _(
    NAMES,
    mo,
    np,
    plt,
    run_mle,
    s5_lam,
    s5_muJ,
    s5_run,
    s5_seed,
    s5_sigJ,
    s5_sigma,
    s5_years,
    simulate_returns,
):
    mo.stop(not s5_run.value, mo.md("*Pulsa **Estimar** para correr la calibración.*"))

    _theta0 = np.array([s5_sigma.value, s5_lam.value, s5_muJ.value, s5_sigJ.value])
    _n = int(252 * s5_years.value)
    mle_x = simulate_returns(_n, _theta0, seed=s5_seed.value)

    _starts = [
        _theta0,
        [0.25, 2.0, -0.01, 0.02],
        [0.15, 10.0, -0.02, 0.03],
        [0.30, 1.0, 0.00, 0.10],
        [0.10, 15.0, -0.05, 0.05],
    ]
    with mo.status.spinner(title="Optimizando desde 5 puntos iniciales..."):
        _best, _fits, mle_se, _cond = run_mle(mle_x, _starts)
    mle_theta = _best.x

    _skew = ((mle_x - mle_x.mean()) ** 3).mean() / mle_x.std() ** 3
    _kurt = ((mle_x - mle_x.mean()) ** 4).mean() / mle_x.std() ** 4 - 3

    _rows = "\n".join(
        f"| {NAMES[i]} | {_theta0[i]:.4f} | {mle_theta[i]:.4f} | {mle_se[i]:.4f} | {mle_se[i] / abs(mle_theta[i]):.0%} |"
        for i in range(4)
    )
    _starts_tbl = "\n".join(
        f"| {i + 1} | {np.round(s, 3).tolist()} | {r.fun:.3f} | {np.round(r.x, 4).tolist()} |"
        for i, (s, r) in enumerate(zip(_starts, _fits))
    )

    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.hist(mle_x, bins=150, density=True, alpha=0.4, color="gray")
    _ax.set_yscale("log")
    _ax.set_ylim(1e-2, None)
    _ax.set_xlabel("retorno diario")
    _ax.set_title(f"Muestra: asimetría {_skew:.2f}, exceso de curtosis {_kurt:.2f}")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                rf"""
    {_n} retornos diarios · $-\ell(\hat\theta)={_best.fun:.3f}$ · número de condición de la Hessiana ≈ ${_cond:.1e}$

    | Parámetro | verdadero | MLE | error est. | error relativo |
    |---|---|---|---|---|
    {_rows}

    **Inicios múltiples** (varios llegan al mismo óptimo; si alguno se detiene antes, el valle es **plano**)

    | # | inicio | −ℓ | solución |
    |---|---|---|---|
    {_starts_tbl}
    """
            ),
            _fig,
        ]
    )
    return mle_theta, mle_x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Resultado 6 · Identificación débil de $\lambda$

    **Verosimilitud perfilada:** para cada $\lambda$ fijo se reoptimizan $(\sigma,\mu_J,\sigma_J)$.
    El IC al 95 % es $\{\lambda:\ \ell(\hat\theta)-\ell_p(\lambda)\le\chi^2_{1,0.95}/2\approx1.92\}$.

    - Curva muy plana: el IC al 95 % de $\lambda$ abarca ≈ $[3,5]$ (error relativo 30–40 %).
    - $\sigma$ y $\sigma_J$: error relativo < 10 %.
    - Causa: con $\lambda\Delta\approx0.02$ solo ~2 % de los días tienen salto, y un salto pequeño se parece a un día de volatilidad alta.

    *(Requiere haber corrido el Resultado 5.)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    s6_run = mo.ui.run_button(label="Calcular perfil de λ")
    s6_run
    return (s6_run,)


@app.cell
def _(fit, mle_theta, mle_x, mo, np, plt, s6_run):
    mo.stop(not s6_run.value, mo.md("*Pulsa el botón.*"))

    _lam_hat = mle_theta[1]
    _grid = np.linspace(max(0.5, 0.25 * _lam_hat), 2.2 * _lam_hat, 22)
    _prof = []
    with mo.status.spinner(title="Perfilando λ..."):
        for _l in _grid:
            _prof.append(fit(mle_x, mle_theta, fixed_lam=_l).fun)
    _prof = np.array(_prof)
    _min = min(_prof.min(), fit(mle_x, mle_theta).fun)
    _delta = _prof - _min

    _inside = _grid[_delta <= 1.92]
    _ci = (_inside.min(), _inside.max()) if len(_inside) else (np.nan, np.nan)

    _fig, _ax = plt.subplots()
    _ax.plot(_grid, _delta, "o-")
    _ax.axhline(1.92, color="C3", ls="--", label="umbral IC 95 % (1.92)")
    _ax.axvline(_lam_hat, color="gray", lw=0.8)
    _ax.set_xlabel("λ")
    _ax.set_ylabel(r"$\ell(\hat\theta)-\ell_p(\lambda)$")
    _ax.set_title(f"Perfil de λ · IC 95 % aprox. [{_ci[0]:.1f}, {_ci[1]:.1f}] (resolución de la malla)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 8 · El "valle plano": superficie de verosimilitud

    Corte de $-\ell$ en el plano de **dos** parámetros (los otros dos fijos en la MLE; *no* es el perfil).
    Las curvas de nivel marcan $\Delta\ell=1.92$ (IC 95 % conjunto aproximado), 5, 10, … Un valle
    **alargado** significa que dos parámetros se compensan entre sí: más saltos pequeños ≈ menos saltos grandes.
    La estrella es la MLE y la cruz roja la verdad. *(Requiere el Resultado 5.)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sf_pair = mo.ui.dropdown(
        {"λ vs σ_J": (1, 3), "λ vs μ_J": (1, 2), "σ vs σ_J": (0, 3), "μ_J vs σ_J": (2, 3)},
        value="λ vs σ_J",
        label="plano",
    )
    sf_pair
    return (sf_pair,)


@app.cell
def _(NAMES, mle_theta, mle_x, nll, np, plt, s5_lam, s5_muJ, s5_sigJ, s5_sigma, sf_pair):
    _i, _j = sf_pair.value
    _hat = np.asarray(mle_theta, float)
    _truth = np.array([s5_sigma.value, s5_lam.value, s5_muJ.value, s5_sigJ.value])

    def _axis(idx, h):
        if idx == 0:
            return h * np.linspace(0.8, 1.2, 21)
        if idx == 1:
            return h * np.linspace(0.3, 2.5, 21)
        if idx == 2:
            return h + np.linspace(-0.05, 0.05, 21)
        return h * np.linspace(0.5, 1.8, 21)

    _a, _b = _axis(_i, _hat[_i]), _axis(_j, _hat[_j])
    _Z = np.zeros((len(_b), len(_a)))
    for _p, _vb in enumerate(_b):
        for _q, _va in enumerate(_a):
            _th = _hat.copy()
            _th[_i], _th[_j] = _va, _vb
            _Z[_p, _q] = nll(_th, mle_x)
    _Z = np.minimum(_Z - _Z.min(), 200)

    _fig, _ax = plt.subplots(figsize=(7, 4.4))
    _cs = _ax.contourf(_a, _b, _Z, levels=[0, 1.92, 5, 10, 20, 50, 100, 200], cmap="viridis_r")
    _ax.contour(_a, _b, _Z, levels=[1.92], colors="w", linewidths=1.6)
    _ax.plot(_hat[_i], _hat[_j], "k*", ms=14, label="MLE")
    _ax.plot(_truth[_i], _truth[_j], "rx", ms=10, mew=2, label="verdad")
    _fig.colorbar(_cs, ax=_ax, label=r"$\ell(\hat\theta)-\ell$")
    _ax.set_xlabel(NAMES[_i])
    _ax.set_ylabel(NAMES[_j])
    _ax.legend()
    _ax.grid(False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 9 · ¿Y si repetimos el experimento? Distribución de los estimadores

    Simulamos **muchas muestras** con los parámetros verdaderos del Resultado 5, calibramos cada una
    (un solo punto inicial, para ir rápido) y dibujamos el histograma de cada estimador.
    $\sigma$ y $\sigma_J$ quedan muy concentrados; $\lambda$ se dispersa: eso **es** la identificación débil.
    Prueba con 6 años contra 30 años de datos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mc_reps = mo.ui.slider(5, 60, step=5, value=20, label="repeticiones")
    mc_years = mo.ui.slider(2, 30, step=1, value=10, label="años por muestra")
    mc_run = mo.ui.run_button(label="Correr experimento")
    mo.hstack([mc_reps, mc_years, mc_run], justify="start")
    return mc_reps, mc_run, mc_years


@app.cell
def _(
    NAMES, fit, mc_reps, mc_run, mc_years, mo, np, plt,
    s5_lam, s5_muJ, s5_sigJ, s5_sigma, simulate_returns,
):
    mo.stop(not mc_run.value, mo.md("*Pulsa **Correr experimento**.*"))

    _theta0 = np.array([s5_sigma.value, s5_lam.value, s5_muJ.value, s5_sigJ.value])
    _n = int(252 * mc_years.value)
    _est = []
    for _s in mo.status.progress_bar(range(mc_reps.value), title="Calibrando muestras..."):
        _x = simulate_returns(_n, _theta0, seed=1000 + _s)
        _est.append(fit(_x, [0.2, 3.0, -0.02, 0.03]).x)
    _est = np.array(_est)

    _fig, _axs = plt.subplots(1, 4, figsize=(12, 3))
    for _p in range(4):
        _axs[_p].hist(_est[:, _p], bins=12, color="C0", alpha=0.7)
        _axs[_p].axvline(_theta0[_p], color="r", lw=2)
        _axs[_p].set_title(NAMES[_p])
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {NAMES[p]} | {_theta0[p]:.4f} | {_est[:, p].mean():.4f} | {_est[:, p].std():.4f} | {_est[:, p].std() / abs(_theta0[p]):.0%} |"
        for p in range(4)
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
    | parámetro | verdadero | media de la MLE | desv. est. | desv./verdadero |
    |---|---|---|---|---|
    {_rows}
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 10 · ¿Se pueden "ver" los saltos? Detección por umbral

    Una idea ingenua: marcar como salto todo día con $|x_t-\text{mediana}|>k\,\hat\sigma\sqrt\Delta$
    ($\hat\sigma$ robusta por MAD). Como **conocemos la verdad** (los días con $N>0$), medimos aciertos y falsas alarmas.
    Con saltos pequeños ($\sigma_J\approx0.04$) la detección falla: ese es el mecanismo detrás de la identificación débil.
    Usa los parámetros verdaderos y la semilla del Resultado 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dj_k = mo.ui.slider(1.5, 6.0, step=0.25, value=3.0, label="umbral k (en σ̂√Δ)")
    dj_k
    return (dj_k,)


@app.cell
def _(
    DT, comp_m, dj_k, mo, norm, np, plt,
    s5_lam, s5_muJ, s5_seed, s5_sigJ, s5_sigma, s5_years,
):
    _sg, _lam, _mu, _sj = s5_sigma.value, s5_lam.value, s5_muJ.value, s5_sigJ.value
    _n = int(252 * s5_years.value)
    _rng = np.random.default_rng(s5_seed.value + 500)
    _N = _rng.poisson(_lam * DT, _n)
    _J = _rng.normal(_mu * _N, _sj * np.sqrt(_N))
    _x = (0.02 - _lam * comp_m(_mu, _sj) - 0.5 * _sg**2) * DT + _sg * np.sqrt(DT) * _rng.standard_normal(_n) + _J

    _med = np.median(_x)
    _sig_hat = 1.4826 * np.median(np.abs(_x - _med)) / np.sqrt(DT)      # σ robusta (MAD)
    _thr = dj_k.value * _sig_hat * np.sqrt(DT)
    _flag = np.abs(_x - _med) > _thr
    _true = _N > 0

    _tp = int((_flag & _true).sum())
    _fp = int((_flag & ~_true).sum())
    _fn = int((~_flag & _true).sum())
    _recall = _tp / max(_tp + _fn, 1)
    _prec = _tp / max(_tp + _fp, 1)
    _fp_null = 2 * (1 - norm.cdf(dj_k.value)) * _n

    _fig, _ax = plt.subplots(figsize=(10, 3.6))
    _ax.plot(_x, ".", ms=2, color="lightgray")
    _ax.plot(np.where(_true)[0], _x[_true], ".", ms=5, color="C0", label="día con salto (verdad)")
    _ax.plot(np.where(_flag)[0], _x[_flag], "o", mfc="none", ms=6, color="C3", label="detectado")
    _ax.axhline(_med + _thr, color="k", ls="--", lw=0.8)
    _ax.axhline(_med - _thr, color="k", ls="--", lw=0.8)
    _ax.set_xlabel("día")
    _ax.set_ylabel("retorno")
    _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
    | días con salto (verdad) | detectados | aciertos | falsas alarmas | recall | precisión | falsas alarmas esperadas sin saltos |
    |---|---|---|---|---|---|---|
    | {int(_true.sum())} | {int(_flag.sum())} | {_tp} | {_fp} | {_recall:.0%} | {_prec:.0%} | {_fp_null:.0f} |
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Tarea 3 (10 min):** estimar sobre los CSV, reportar errores estándar y comparar inicios alternativos.

    **Calibra con tus propios datos.** Sube un CSV con retornos diarios o precios (p. ej. `retornos_diarios_sp500.csv`,
    `retornos_diarios_aapl.csv`). Se usa la columna `ret`/`return`/`close`/`price` si existe; si no, la última numérica.
    La carpeta `data/` del repositorio trae series reales (S&P 500, AAPL y otras) y sintéticas con verdad conocida; ver `data/README.md`.
    """)
    return


@app.cell
def _(csv, np):
    def parse_csv_series(raw):
        """Lee un CSV y devuelve (nombre_columna, valores) de la columna numérica relevante."""
        text = raw.decode("utf-8-sig", errors="ignore")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 3:
            raise ValueError("El archivo tiene muy pocas filas.")
        delim = max([",", ";", "\t"], key=lambda d: lines[0].count(d))
        rows = list(csv.reader(lines, delimiter=delim))

        def to_float(s):
            s = s.strip().replace("%", "")
            if delim == ";":
                s = s.replace(",", ".")
            try:
                return float(s)
            except ValueError:
                return float("nan")

        first_numeric = all(np.isfinite(to_float(c)) for c in rows[0] if c.strip())
        header = [f"col{i}" for i in range(len(rows[0]))] if first_numeric else [c.strip() for c in rows[0]]
        body = rows if first_numeric else rows[1:]
        cols = {}
        for j, name in enumerate(header):
            vals = np.array([to_float(r[j]) if j < len(r) else float("nan") for r in body])
            if np.isfinite(vals).mean() >= 0.9:
                cols[name] = vals[np.isfinite(vals)]
        if not cols:
            raise ValueError("No encontré columnas numéricas.")
        prefer = ["ret", "return", "returns", "retorno", "retornos", "log_ret", "logret",
                  "adj close", "adj_close", "close", "price", "precio", "cierre"]
        for p in prefer:
            for name in cols:
                if name.lower() == p:
                    return name, cols[name]
        name = list(cols)[-1]
        return name, cols[name]

    return (parse_csv_series,)


@app.cell(hide_code=True)
def _(mo):
    up_file = mo.ui.file(filetypes=[".csv", ".txt"], kind="area", label="Arrastra aquí tu CSV")
    up_kind = mo.ui.radio(
        options={"Retornos logarítmicos": "log", "Retornos simples": "simple", "Precios": "price"},
        value="Retornos logarítmicos",
        label="Los datos son",
    )
    up_alpha = mo.ui.slider(-0.1, 0.2, step=0.005, value=0.02, label="α (deriva fija)")
    up_run = mo.ui.run_button(label="Calibrar")
    mo.vstack([up_file, mo.hstack([up_kind, up_alpha], justify="start"), up_run])
    return up_alpha, up_file, up_kind, up_run


@app.cell
def _(
    BOUNDS,
    NAMES,
    mo,
    np,
    parse_csv_series,
    plt,
    run_mle,
    up_alpha,
    up_file,
    up_kind,
    up_run,
):
    mo.stop(len(up_file.value) == 0, mo.md("*Sube un CSV para calibrar.*"))
    mo.stop(not up_run.value, mo.md("*Pulsa **Calibrar**.*"))

    try:
        _col, _raw = parse_csv_series(up_file.contents())
    except Exception as _e:
        mo.stop(True, mo.md(f"**No pude leer el archivo:** {_e}"))

    if up_kind.value == "price":
        _x = np.diff(np.log(_raw[_raw > 0]))
    elif up_kind.value == "simple":
        _x = np.log1p(_raw[_raw > -1])
    else:
        _x = _raw
    _x = _x[np.isfinite(_x)]
    mo.stop(len(_x) < 250, mo.md("**Muy pocas observaciones** (se necesitan al menos 250)."))

    _s0 = float(np.std(_x) * np.sqrt(252))
    _starts = [
        [0.8 * _s0, 5.0, -0.02, 0.03],
        [_s0, 2.0, -0.01, 0.02],
        [0.6 * _s0, 10.0, -0.03, 0.04],
        [_s0, 1.0, 0.00, 0.08],
        [0.5 * _s0, 15.0, -0.05, 0.05],
    ]
    with mo.status.spinner(title="Optimizando desde 5 puntos iniciales..."):
        _best, _fits, _se, _cond = run_mle(_x, _starts, alpha=up_alpha.value)

    _rows = "\n".join(
        f"| {NAMES[i]} | {_best.x[i]:.4f} | {_se[i]:.4f} | {_se[i] / abs(_best.x[i]):.0%} |" for i in range(4)
    )
    _starts_tbl = "\n".join(
        f"| {i + 1} | {r.fun:.3f} | {np.round(r.x, 4).tolist()} |" for i, r in enumerate(_fits)
    )
    _skew = ((_x - _x.mean()) ** 3).mean() / _x.std() ** 3
    _kurt = ((_x - _x.mean()) ** 4).mean() / _x.std() ** 4 - 3

    _edge = [
        NAMES[i]
        for i, (_lo, _hi) in enumerate(BOUNDS)
        if min(abs(_best.x[i] - _lo), abs(_best.x[i] - _hi)) < 1e-4 * max(abs(_hi), 1.0)
    ]
    _warn = (
        mo.callout(
            mo.md(
                f"**El óptimo está en la cota de {', '.join(_edge)}.** Los errores estándar de la Hessiana "
                "no son válidos en un borde. Con datos diarios reales es habitual: la mezcla usa muchos saltos "
                "diminutos (λ grande, σ_J pequeño) para imitar las colas pesadas, en lugar de pocos saltos grandes."
            ),
            kind="warn",
        )
        if _edge
        else mo.md("")
    )

    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.hist(_x, bins=150, density=True, alpha=0.4, color="gray")
    _ax.set_yscale("log")
    _ax.set_ylim(1e-2, None)
    _ax.set_xlabel("retorno diario")
    _ax.set_title(f"Columna «{_col}»: asimetría {_skew:.2f}, exceso de curtosis {_kurt:.2f}")
    _fig.tight_layout()

    mo.vstack(
        [
            _warn,
            mo.md(
                rf"""
    {len(_x)} retornos · $-\ell(\hat\theta)={_best.fun:.3f}$ · número de condición de la Hessiana ≈ ${_cond:.1e}$

    | Parámetro | MLE | error est. | error relativo |
    |---|---|---|---|
    {_rows}

    | # | −ℓ | solución |
    |---|---|---|
    {_starts_tbl}
    """
            ),
            _fig,
        ]
    )
    return


# ----------------------------------------------------------------------
# Smile
# ----------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # De la calibración a la decisión: smile

    Se valora con la fórmula de Merton y se invierte Black–Scholes con `brentq`
    (`brentq(lambda s: bs_call(S0,K,T,r,s) - price, 1e-4, 5.0)`; aquí con la opción OTM por estabilidad numérica;
    es la función `merton_iv` de la Hora 1).

    Por defecto: $S_0=100$, $r=0.03$, $\sigma=0.15$, $\lambda=0.6$, $\mu_J=-0.15$, $\sigma_J=0.12$. En la diapositiva, IV (%) en $\ln(K/S_0)=\mp0.21$:

    | $T$ | put OTM | ATM | call OTM |
    |---|---|---|---|
    | 0.1 | 35.8 | 17.9 | 20.0 |
    | 0.5 | 24.4 | 19.7 | 17.5 |
    | 1.0 | 22.5 | 20.3 | 18.5 |
    | 2.0 | 21.6 | 20.5 | 19.5 |

    > **Comité de riesgo:** ¿en qué dirección y magnitud corrige BS la presencia de saltos y qué parte del smile captura el modelo?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    s7_sigma = mo.ui.slider(0.05, 0.4, step=0.01, value=0.15, label="σ")
    s7_lam = mo.ui.slider(0.0, 5.0, step=0.1, value=0.6, label="λ")
    s7_muJ = mo.ui.slider(-0.4, 0.2, step=0.01, value=-0.15, label="μ_J")
    s7_sigJ = mo.ui.slider(0.01, 0.4, step=0.01, value=0.12, label="σ_J")
    s7_r = mo.ui.slider(0.0, 0.1, step=0.005, value=0.03, label="r")
    mo.vstack(
        [
            mo.hstack([s7_sigma, s7_lam, s7_r], justify="start"),
            mo.hstack([s7_muJ, s7_sigJ], justify="start"),
        ]
    )
    return s7_lam, s7_muJ, s7_r, s7_sigJ, s7_sigma


@app.cell
def _(math, merton_iv, mo, np, plt, s7_lam, s7_muJ, s7_r, s7_sigJ, s7_sigma):
    _S0 = 100.0
    _r, _sg, _lam, _mu, _sj = s7_r.value, s7_sigma.value, s7_lam.value, s7_muJ.value, s7_sigJ.value

    def _iv(K, T):
        return merton_iv(_S0, K, T, _r, _sg, _lam, _mu, _sj)

    _Ts = [0.1, 0.5, 1.0, 2.0]
    _xs = np.linspace(-0.4, 0.4, 41)
    _fig, _ax = plt.subplots()
    for _T in _Ts:
        _ivs = [100 * _iv(_S0 * math.exp(x), _T) for x in _xs]
        _ax.plot(_xs, _ivs, label=f"T = {_T}")
    _ax.set_xlabel(r"$\ln(K/S_0)$")
    _ax.set_ylabel("IV (%)")
    _ax.legend()
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {_T} | {100 * _iv(_S0 * math.exp(-0.21), _T):.1f} | {100 * _iv(_S0, _T):.1f} | {100 * _iv(_S0 * math.exp(0.21), _T):.1f} |"
        for _T in _Ts
    )
    _lim = 100 * math.sqrt(_sg**2 + _lam * (_sj**2 + _mu**2))
    _atm_long = 100 * _iv(_S0, 30.0)
    _tbl = mo.md(
        rf"""
    IV (%) en $\ln(K/S_0)=\mp0.21$

    | T | put OTM | ATM | call OTM |
    |---|---|---|---|
    {_rows}

    IV ATM a $T=30$: {_atm_long:.1f} % · límite heurístico $\sqrt{{\sigma^2+\lambda(\sigma_J^2+\mu_J^2)}}$: {_lim:.1f} %
    """
    )
    mo.hstack([_fig, _tbl], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 11 · Estructura temporal del skew

    Con los parámetros del smile (arriba), medimos el **skew** $\text{IV}(\ln K/S_0=-0.2)-\text{IV}(+0.2)$
    para muchos plazos. La teoría dice que decae como $T^{-1/2}$ a plazos largos (línea punteada);
    el dato empírico decae **más lento**: es el límite del modelo.
    A la derecha, la IV ATM converge a $\sqrt{\sigma^2+\lambda(\sigma_J^2+\mu_J^2)}$.
    """)
    return


@app.cell
def _(math, merton_iv, np, plt, s7_lam, s7_muJ, s7_r, s7_sigJ, s7_sigma):
    _S0 = 100.0
    _r, _sg, _lam, _mu, _sj = s7_r.value, s7_sigma.value, s7_lam.value, s7_muJ.value, s7_sigJ.value
    _Ts = np.logspace(math.log10(0.1), math.log10(15), 14)
    _put = np.array([merton_iv(_S0, _S0 * math.exp(-0.2), T, _r, _sg, _lam, _mu, _sj) for T in _Ts])
    _call = np.array([merton_iv(_S0, _S0 * math.exp(0.2), T, _r, _sg, _lam, _mu, _sj) for T in _Ts])
    _atm = np.array([merton_iv(_S0, _S0, T, _r, _sg, _lam, _mu, _sj) for T in _Ts])
    _skew = _put - _call
    _lim = math.sqrt(_sg**2 + _lam * (_sj**2 + _mu**2))

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8))
    _axs[0].loglog(_Ts, np.abs(_skew), "o-", label="skew simulado")
    _i = len(_Ts) // 2
    _axs[0].loglog(_Ts[_i:], abs(_skew[_i]) * (_Ts[_i:] / _Ts[_i]) ** -0.5, "k:", label=r"pendiente $-1/2$")
    _axs[0].set_xlabel("T")
    _axs[0].set_title("|IV(−0.2) − IV(+0.2)|")
    _axs[0].legend()

    _axs[1].semilogx(_Ts, 100 * _atm, "o-", label="IV ATM")
    _axs[1].axhline(100 * _lim, color="r", ls="--", label="límite TCL")
    _axs[1].set_xlabel("T")
    _axs[1].set_ylabel("IV (%)")
    _axs[1].legend()
    _fig.tight_layout()
    _fig
    return


# ----------------------------------------------------------------------
# Cobertura
# ----------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Cobertura: mercado incompleto

    $d\Pi=\Delta\,dS+dC-r(\Delta S-C)\,dt$. El término de salto es

    $$\big[\Delta S(Y-1)+C(SY)-C(S)\big]\,dN_t,$$

    que **no se anula** para ningún $\Delta$ determinista: $C(SY)-C(S)$ no es lineal en $Y$. El delta-hedge anula el riesgo difusivo pero deja **riesgo de salto residual**.

    > **Prueba corta.** Anularlo exigiría $\Delta S(y-1)=C(S)-C(Sy)$ para todo $y\in(0,\infty)$, es decir, $C$ lineal en $S$. La call es convexa. No existe cartera replicante.

    El gráfico siguiente muestra el residuo $R(y)=\Delta S(y-1)+C(Sy)-C(S)$ para el delta del modelo y otros valores de $\Delta$: ninguno lo anula.

    ---
    # Resultado 7 · Cobertura delta simulada

    Vendedor de una call ATM, $T=0.5$, $r=0.03$, 126 rebalanceos, 6000 trayectorias, bajo $\mathbb Q$. Merton: $\sigma=0.15$, $\lambda=1$, $\mu_J=-0.15$, $\sigma_J=0.25$. Referencia: GBM $\sigma=0.20$ (delta de BS).

    En la diapositiva: P&L medio $-0.01$ vs. $0.00$; desv. est. $7.60$ vs. $0.44$; cuantil 1 % $-28.4$ vs. $-1.2$; asimetría $-2.50$.
    Media ≈ 0, pero **cola izquierda enorme**: acciones no cubren el riesgo de salto.

    **Del papel al código.** `delta_hedge_pnl` implementa la estrategia autofinanciada: cada día se
    recalcula el delta de Merton con `merton_terms`, se ajusta la posición en la acción con caja que
    capitaliza a tasa $r$, y al vencimiento se compara con el payoff de la call vendida.
    """)
    return


@app.cell
def _(merton_terms, np, simulate_merton):
    def delta_hedge_pnl(S0, K, r, T, n, n_paths, sigma, lam, muJ, sigJ, seed=0):
        """P&L del vendedor de una call con delta-hedge discreto (autofinanciado)."""
        dt = T / n
        S, _ = simulate_merton(S0, r, sigma, lam, muJ, sigJ, T, n, n_paths, seed=seed)
        C0, d0, _, _ = merton_terms(np.array([S0]), K, T, r, sigma, lam, muJ, sigJ)
        delta = np.full(n_paths, float(d0[0]))
        cash = float(C0[0]) - delta * S0                       # prima cobrada − compra de Δ acciones
        for j in range(1, n + 1):
            cash = cash * np.exp(r * dt)                       # la caja rinde r
            if j < n:
                _, dj, _, _ = merton_terms(S[:, j], K, T - j * dt, r, sigma, lam, muJ, sigJ)
                cash = cash - (dj - delta) * S[:, j]           # rebalanceo
                delta = dj
        return cash + delta * S[:, -1] - np.maximum(S[:, -1] - K, 0.0)

    return (delta_hedge_pnl,)


@app.cell(hide_code=True)
def _(mo):
    s8_T = mo.ui.slider(0.1, 2.0, step=0.1, value=0.5, label="T")
    s8_nreb = mo.ui.slider(12, 504, step=6, value=126, label="rebalanceos")
    s8_sigma = mo.ui.slider(0.05, 0.4, step=0.01, value=0.15, label="σ")
    s8_lam = mo.ui.slider(0.0, 5.0, step=0.1, value=1.0, label="λ")
    s8_muJ = mo.ui.slider(-0.4, 0.2, step=0.01, value=-0.15, label="μ_J")
    s8_sigJ = mo.ui.slider(0.01, 0.5, step=0.01, value=0.25, label="σ_J")
    s8_paths = mo.ui.dropdown(
        {"2000": 2000, "6000": 6000, "20000": 20000}, value="6000", label="trayectorias"
    )
    s8_run = mo.ui.run_button(label="Simular cobertura")
    mo.vstack(
        [
            mo.hstack([s8_T, s8_nreb, s8_paths], justify="start"),
            mo.hstack([s8_sigma, s8_lam, s8_muJ, s8_sigJ], justify="start"),
            s8_run,
        ]
    )
    return (
        s8_T,
        s8_lam,
        s8_muJ,
        s8_nreb,
        s8_paths,
        s8_run,
        s8_sigJ,
        s8_sigma,
    )


@app.cell
def _(merton_terms, np, plt, s8_T, s8_lam, s8_muJ, s8_sigJ, s8_sigma):
    _S0, _K, _r = 100.0, 100.0, 0.03
    _T = s8_T.value
    _sg, _lam, _mu, _sj = s8_sigma.value, s8_lam.value, s8_muJ.value, s8_sigJ.value

    _C0, _d0, _, _ = merton_terms(np.array(_S0), _K, _T, _r, _sg, _lam, _mu, _sj)
    _y = np.linspace(0.5, 1.5, 200)
    _Cy, _, _, _ = merton_terms(_S0 * _y, _K, _T, _r, _sg, _lam, _mu, _sj)

    _fig, _ax = plt.subplots()
    for _d, _lab in [(float(_d0), "Δ del modelo"), (0.3, "Δ = 0.3"), (0.8, "Δ = 0.8")]:
        _ax.plot(_y, _d * _S0 * (_y - 1) + _Cy - float(_C0), label=f"{_lab} ({_d:.2f})")
    _ax.axhline(0, color="k", lw=0.8)
    _ax.set_xlabel("factor de salto y = Y")
    _ax.set_ylabel(r"$R(y)=\Delta S(y-1)+C(Sy)-C(S)$")
    _ax.set_title("Residuo de salto del delta-hedge (call ATM)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    delta_hedge_pnl,
    mo,
    np,
    plt,
    s8_T,
    s8_lam,
    s8_muJ,
    s8_nreb,
    s8_paths,
    s8_run,
    s8_sigJ,
    s8_sigma,
):
    mo.stop(not s8_run.value, mo.md("*Pulsa **Simular cobertura**.*"))

    with mo.status.spinner(title="Simulando cobertura (Merton y GBM)..."):
        _pm = delta_hedge_pnl(
            100.0, 100.0, 0.03, s8_T.value, s8_nreb.value, s8_paths.value,
            s8_sigma.value, s8_lam.value, s8_muJ.value, s8_sigJ.value, seed=21,
        )
        _pg = delta_hedge_pnl(
            100.0, 100.0, 0.03, s8_T.value, s8_nreb.value, s8_paths.value,
            0.20, 0.0, 0.0, 0.01, seed=22,
        )

    def _skew(x):
        return ((x - x.mean()) ** 3).mean() / x.std() ** 3

    _fig, _axs = plt.subplots(1, 2, figsize=(10, 3.8))
    _lo = min(np.quantile(_pm, 0.001), np.quantile(_pg, 0.001))
    _hi = max(np.quantile(_pm, 0.999), np.quantile(_pg, 0.999))
    _bins = np.linspace(_lo, _hi, 120)
    _axs[0].hist(_pg, bins=_bins, alpha=0.6, label="GBM (BS)", density=True)
    _axs[0].hist(_pm, bins=_bins, alpha=0.6, label="Merton", density=True)
    _axs[0].set_xlabel("P&L del vendedor")
    _axs[0].legend()
    _axs[1].hist(_pg, bins=_bins, alpha=0.6, density=True)
    _axs[1].hist(_pm, bins=_bins, alpha=0.6, density=True)
    _axs[1].set_yscale("log")
    _axs[1].set_xlabel("P&L (escala log)")
    _fig.tight_layout()

    _tbl = mo.md(
        rf"""
    | P&L | Merton | GBM |
    |---|---|---|
    | media | {_pm.mean():.2f} | {_pg.mean():.2f} |
    | desv. est. | {_pm.std():.2f} | {_pg.std():.2f} |
    | cuantil 1 % | {np.quantile(_pm, 0.01):.1f} | {np.quantile(_pg, 0.01):.1f} |
    | asimetría | {_skew(_pm):.2f} | {_skew(_pg):.2f} |
    """
    )
    mo.vstack([_fig, _tbl])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 12 · ¿Cubrir más seguido ayuda? Frecuencia de rebalanceo

    En BS, rebalancear más seguido hace tender el riesgo a 0. Con saltos, **no**: el error se estanca
    en un piso, porque el salto llega de golpe y ningún rebalanceo lo anticipa.
    Usa los parámetros de Merton del Resultado 7 (deslizadores de arriba).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    hf_paths = mo.ui.dropdown({"1000": 1000, "2000": 2000, "4000": 4000}, value="2000", label="trayectorias")
    hf_run = mo.ui.run_button(label="Barrer frecuencias")
    mo.hstack([hf_paths, hf_run], justify="start")
    return hf_paths, hf_run


@app.cell
def _(
    delta_hedge_pnl, hf_paths, hf_run, mo, np, plt,
    s8_T, s8_lam, s8_muJ, s8_sigJ, s8_sigma,
):
    mo.stop(not hf_run.value, mo.md("*Pulsa **Barrer frecuencias**.*"))

    _freqs = [4, 12, 26, 52, 126, 252]
    _sd_m, _sd_g, _q_m, _q_g = [], [], [], []
    for _nr in mo.status.progress_bar(_freqs, title="Simulando coberturas..."):
        _pm = delta_hedge_pnl(100.0, 100.0, 0.03, s8_T.value, _nr, hf_paths.value,
                              s8_sigma.value, s8_lam.value, s8_muJ.value, s8_sigJ.value, seed=31)
        _pg = delta_hedge_pnl(100.0, 100.0, 0.03, s8_T.value, _nr, hf_paths.value,
                              0.20, 0.0, 0.0, 0.01, seed=32)
        _sd_m.append(_pm.std())
        _sd_g.append(_pg.std())
        _q_m.append(np.quantile(_pm, 0.01))
        _q_g.append(np.quantile(_pg, 0.01))

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8))
    _axs[0].loglog(_freqs, _sd_m, "o-", label="Merton")
    _axs[0].loglog(_freqs, _sd_g, "s-", label="GBM (BS)")
    _axs[0].set_xlabel("número de rebalanceos")
    _axs[0].set_ylabel("desv. est. del P&L")
    _axs[0].legend()
    _axs[1].semilogx(_freqs, _q_m, "o-", label="Merton")
    _axs[1].semilogx(_freqs, _q_g, "s-", label="GBM (BS)")
    _axs[1].set_xlabel("número de rebalanceos")
    _axs[1].set_ylabel("cuantil 1 % del P&L")
    _axs[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Elección de $\mathbb Q$ y griegas

    **Elección de $\mathbb Q$.** El conjunto de medidas martingala equivalentes es infinito. Criterios: mínima entropía, Esscher, utilidad exponencial, varianza mínima.

    - **Merton:** con riesgo de salto diversificable, los parámetros de salto bajo $\mathbb P$ y $\mathbb Q$ coinciden.
    - **Índices en la práctica:** hay prima por riesgo de salto, así que $\lambda^{\mathbb Q}\ne\lambda^{\mathbb P}$. Calibrar con retornos ($\mathbb P$) y con opciones ($\mathbb Q$) da parámetros distintos: de ahí la tarea de extensión.

    **Griegas.** $\Delta=\partial C/\partial S$ existe pero no cubre saltos; el vega respecto a $\sigma$ difiere del de BS y aparecen sensibilidades a $\lambda,\mu_J,\sigma_J$. Cálculo por diferencias finitas o *pathwise*.

    ---
    # Ticket de salida (individual, 3 min)

    1. Escribe la fórmula de Merton e identifica el factor $e^{(r_k-r)T}$ (o, equivalente, $\lambda'$). ¿Qué pasaría si se omitiera?
    2. **Análisis.** Dos acciones con igual IV ATM y distinta curvatura del smile: ¿cuál tiene mayor $\lambda$ y por qué?
    3. **Transferencia.** Si el salto fuera exponencial doble (Kou), ¿qué cumulantes cambian y cómo afecta al smile?
    4. **Concepto.** En una línea: ¿por qué falla la cobertura perfecta bajo MJD?

    > **Metacognición (2 min).** *"El paso del problema que aún no podría explicar sin ayuda es…"*

    ---
    # Tarea de extensión y evaluación

    **Vía B.** Calibrar por mínimos cuadrados sobre precios de opciones (SPX o una acción del índice) usando $\phi_T$ y Carr–Madan, y comparar con la MLE. **Entregable:** notebook con código, tabla de parámetros y gráfico de smile ajustado vs. empírico.

    | Componente | Puntos | Criterios clave |
    |---|---|---|
    | Ticket de salida | 4 | Factor $e^{(r_k-r)T}$, $\lambda$ vs. curvatura, Kou, incompletitud |
    | Calibración en parejas | 6 | Código válido (2), errores estándar (1), identificación débil (2), inicios alternativos (1) |
    | Extensión Vía B | 10 | Ajuste (3), comparación con MLE (2), gráfico (2), discusión de límites (3) |

    *(Punto de partida para la Vía B: `merton_cf` y `carr_madan_call` están definidas en la Hora 1 de este cuaderno.)*

    ---
    # Síntesis

    - Merton = **difusión + Poisson compuesto**; el compensador $\lambda m$ hace de $e^{-rt}S_t$ una martingala, y $\ln S$ es un proceso de Lévy de actividad finita.
    - Precio = **mezcla Poisson de BS**: $C_0=\sum_kP_k'\,\mathrm{BS}(r_k,\sigma_k)$, con error de truncamiento acotado por $S_0\,\mathbb P(N'>K^*)$.
    - Cumulantes $\propto T$ ⇒ skew y colas que **decaen como $T^{-1/2}$ y $T^{-1}$**.
    - **Calibrar es difícil:** $\lambda$ apenas identificado con datos diarios; la MLE converge al mejor ajuste en KL.
    - **Mercado incompleto:** el delta-hedge deja una cola izquierda de riesgo de salto.
    - Siguiente paso: Kou, Bates y calibración por Fourier.
    """)
    return


if __name__ == "__main__":
    app.run()
