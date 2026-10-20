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
# Portada, objetivos, ruta, notación
# ======================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **UNIVERSIDAD EXTERNADO DE COLOMBIA**

    # Modelo de saltos por difusión de Merton (1976)

    ## Contexto, formalización matemática, simulación y calibración

    Acciones e índices bursátiles · Clase de 120 min (60 teoría + 60 simulación) · Definiciones, teoremas, corolarios, ejemplos y ejercicios

    *Este cuaderno contiene la presentación completa, con **la misma estructura y numeración** que las
    diapositivas (Definición 1…6, Teorema 1…7, Ejemplo 1…8, Ejercicio 1…9). Cada bloque de teoría va seguido
    de la **celda de código** que lo implementa y de un **laboratorio interactivo**: mueve los
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
    # Objetivos de aprendizaje

    Al terminar la sesión, el estudiante podrá:

    1. **Especificar** la EDS de Merton, resolverla con Itô con saltos y **demostrar** la martingala que impone el compensador $\lambda m$.
    2. **Derivar** el precio de una opción europea como mezcla Poisson de precios Black–Scholes ($r_k$, $\sigma_k$).
    3. **Acotar** el error de truncamiento con una cota explícita de cola de Poisson y **verificar** contra Monte Carlo.
    4. **Calibrar** $(\sigma,\lambda,\mu_J,\sigma_J)$ por MLE y **diagnosticar** identificación débil.
    5. **Explicar** el mercado incompleto, el fallo del delta-hedge y los límites del smile.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <span style="background:#444b52;color:#f6efd9;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">RUTA</span>

    # Ruta de la sesión · 120 minutos

    | Hora | Bloque | Min | Contenido | Diapositivas | Código / laboratorio |
    |---|---|---|---|---|---|
    | **1 · Teoría** | 0 | 0–12 | Contexto histórico y motivación empírica | § 0 | — |
    |  | 1 | 12–24 | EDS, solución explícita, martingala | § 1 | `simulate_decomposed` · **Lab 1** |
    |  | 2 | 24–36 | Ley condicional, cumulantes | § 2: Prop. 2, Teo. 3, Cor. 3–4 | **Lab 2** (mezcla) · gráfico de cumulantes |
    |  | 3 | 36–49 | **Fórmula de valoración** y cota de truncamiento | § 3: Teo. 5, Prop. 3, Cor. 8 | `merton_terms` · **Lab 3** (términos) · **Lab 4** (sensibilidad) · **Lab 5** (mapa) |
    |  | 4 | 49–57 | Función característica y smile | Teo. 4 (§ 2); § 4 | `carr_madan_call` |
    |  | 5 | 57–60 | Limitaciones | Limitaciones y extensiones | — |
    | **2 · Simulación** | 1–2 | 0–26 | Simulación exacta; suma truncada vs. Monte Carlo | Sim. 1–2, Ejemplo 4, Ejercicios 3–5 | `simulate_merton` · **Lab 6** (martingala y Poisson) · **Lab 7** (colas, Ejemplo 2) |
    |  | 3 | 26–38 | Smile e inversión de Black–Scholes | Ejemplo 5, Ejercicio 6 | `merton_iv` · **Lab 11** (estructura temporal del skew) |
    |  | 4 | 38–52 | **Calibración por MLE** | § 5, Ejemplos 6–7, Ejercicios 7–8 | `nll`, `fit` · **Lab 8** (superficie) · **Lab 9** (Monte Carlo de estimadores) · **Lab 10** (detección de saltos) |
    |  | 5 | 52–60 | Cobertura, mercado incompleto, ticket de salida | § 6, Ejemplo 8, Ejercicio 9 | `delta_hedge_pnl` · **Lab 12** (frecuencia de rebalanceo) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objetivos y estructura de la sesión

    Al terminar, el estudiante podrá **definir** el modelo con precisión, **demostrar** sus propiedades fundamentales, **simular y calibrar**, **aplicarlas** en ejemplos numéricos y **resolver** ejercicios.

    | § | Tema | Resultados principales | Simulación y aplicación |
    |---|---|---|---|
    | 0 | Contexto y motivación | — | Evidencia empírica en renta variable |
    | 1 | El modelo y su martingala | Def. 1–3, Lema 1, Teo. 1–2, Cor. 1–2, Prop. 1 | Simulación exacta; trayectorias |
    | 2 | Ley y cumulantes | Prop. 2, Teo. 3–4, Cor. 3–4 | Ejemplo 2: momentos |
    | 3 | Valoración de opciones | Lema 2, Teo. 5, Cor. 5–8, Prop. 3 | Suma truncada vs. Monte Carlo |
    | 4 | Volatilidad implícita | Def. 4, Prop. 4 | Ejemplo 5: smile |
    | 5 | Calibración por MLE | Def. 5, Prop. 5, Teo. 6, Cor. 9 | Ejemplos 6–7 |
    | 6 | Cobertura e incompletitud | Def. 6, Teo. 7, Cor. 10 | Ejemplo 8: delta-hedge |

    ---

    # Cómo leer las diapositivas

    Cada diapositiva lleva una **etiqueta de color** según su tipo:

    | Etiqueta | Uso |
    |---|---|
    | **CONTEXTO** (ocre) | Motivación histórica y empírica |
    | **DEFINICIÓN** (verde) | Objetos y supuestos del modelo |
    | **LEMA / PROPOSICIÓN** (azul) | Resultados auxiliares o intermedios |
    | **TEOREMA** (turquesa) | Resultados centrales, con demostración |
    | **COROLARIO** (dorado) | Consecuencias inmediatas de un teorema |
    | **OBSERVACIÓN** (gris verdoso) | Comentarios, límites y contexto |
    | **EJEMPLO** (violeta) | Cálculos trabajados con cifras |
    | **SIMULACIÓN** (naranja) | Código y resultados de la simulación |
    | **EJERCICIO** (rojo) | Para resolver en clase |
    | **SOLUCIÓN** (verde oscuro) | Soluciones de los Ejercicios 1–9, al final |
    | **NOTACIÓN** (azul acero) | Símbolos y convenciones; material de consulta |
    | **GLOSARIO / REFERENCIAS** (grafito) | Siglas, conceptos y fuentes; anexo, de consulta |

    Las demostraciones terminan con $\square$. Las cifras teóricas se calculan con las fórmulas de la sesión; las simuladas provienen de `simulacion_merton.py` (semilla fija).

    ---

    <span style="background:#4b6584;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">NOTACIÓN</span>

    # Notación 1/3 · Probabilidad y procesos

    | Símbolo | Significado |
    |---|---|
    | $\mathbb Q,\ \mathbb P$ | medida neutral al riesgo (§§ 1–4 y 6); medida física, donde viven los datos (§ 5) |
    | $\mathbb E,\ \mathbb P(\cdot)$ | esperanza y probabilidad, bajo $\mathbb Q$ salvo en § 5 (en el modelo de Merton, $N$ y $J$ tienen la misma ley bajo $\mathbb P$ y $\mathbb Q$) |
    | $(\mathcal F_t)$ | filtración: información disponible hasta $t$; *condiciones usuales* = completa y continua por la derecha |
    | $S_{t^-}$ | $\lim_{s\uparrow t}S_s$, valor justo antes de $t$; un integrando **predecible** es adaptado y continuo por la izquierda |
    | $\mathcal N(\mu,s^2)$ | ley normal de media $\mu$ y varianza $s^2$ |
    | $\varphi(x;\mu,s^2)$ | densidad de $\mathcal N(\mu,s^2)$ en $x$: $\frac{1}{\sqrt{2\pi s^2}}e^{-(x-\mu)^2/(2s^2)}$; $\varphi(x)=\varphi(x;0,1)$ |
    | $\Phi$ | función de distribución de $\mathcal N(0,1)$ |
    | $(x)^+,\ \mathbf 1_A$ | $\max(x,0)$; indicadora del evento $A$ (vale 1 si ocurre y 0 si no) |
    | $i,\ \square$ | unidad imaginaria $\sqrt{-1}$ (distinta del índice $i$ de $J_i$); fin de demostración |

    ---

    <span style="background:#4b6584;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">NOTACIÓN</span>

    # Notación 2/3 · Parámetros y variables del modelo

    | Símbolo | Significado |
    |---|---|
    | $S_0,\ T,\ K$ | precio inicial; plazo (años); precio de ejercicio (*strike*) |
    | $r,\ \sigma$ | tasa libre de riesgo (continua, anual); volatilidad anual de la parte difusiva |
    | $\lambda$ | intensidad: número esperado de saltos por año |
    | $\mu_J,\ \sigma_J$ | media y desviación estándar del log-salto $J=\ln Y$ |
    | $m$ | $\mathbb E[Y-1]$: salto relativo medio (compensador) |
    | $\lambda'=\lambda(1+m)$ | intensidad ajustada de la mezcla de BS (Cor. 5) |
    | $k,\ P_k,\ P_k'$ | número de saltos; pesos de Poisson con $\lambda T$ y con $\lambda'T$ |
    | $\mu_k,\ s_k^2$ | media y varianza de $\ln(S_T/S_0)$ dado $N_T=k$ |
    | $r_k,\ \sigma_k$ | tasa y volatilidad efectivas del término $k$; $s_k^2=\sigma_k^2T$ |
    | $K^*,\ N'$ | índice de truncamiento; variable $N'\sim\text{Poisson}(\lambda'T)$ |
    | $\theta,\ \Theta$ | vector $(\sigma,\lambda,\mu_J,\sigma_J)$ y su espacio paramétrico |
    | $\Delta$ | en § 5, paso entre datos (en años; $1/252$ si es diario); en § 6, delta de cobertura $\partial C/\partial S$ |

    ---

    <span style="background:#4b6584;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">NOTACIÓN</span>

    # Notación 3/3 · Black–Scholes y convenciones

    **Precio de Black–Scholes** de una call con tasa $\rho$ y volatilidad $s$ (con $\rho=r$ es el BS usual; en código, `bs_call`):

    $$\mathrm{BS}(S_0,K,T,\rho,s)=S_0\Phi(d_1)-Ke^{-\rho T}\Phi(d_2),\quad d_1=\frac{\ln(S_0/K)+(\rho+s^2/2)T}{s\sqrt T},\quad d_2=d_1-s\sqrt T .$$

    Su **vega** es $\partial\mathrm{BS}/\partial s=S_0\,\varphi(d_1)\sqrt T>0$.

    | Convención | Significado |
    |---|---|
    | ATM / ITM / OTM | *at / in / out of the money*. Call: $K=S_0$ / $K<S_0$ / $K>S_0$. Put: $K=S_0$ / $K>S_0$ / $K<S_0$ |
    | e.e., $\pm$, $z$ | error estándar; la cifra tras $\pm$ es un error estándar; $z=(\text{simulado}-\text{teoría})/\text{e.e.}$ y $\lvert z\rvert<1.96$ no rechaza al 95 % |
    | pp | puntos porcentuales (diferencia entre dos porcentajes) |
    | Código | `np` = NumPy; `norm` = `scipy.stats.norm`; `brentq`, `minimize` = `scipy.optimize` |
    | Datos | `retornos_diarios_*.csv`: una fila por día hábil, con fecha y retorno logarítmico $x_i=\ln(S_{t_i}/S_{t_{i-1}})$; $\Delta=1/252$ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **HORA 1**

    # Teoría

    ---

    # § 0 · Contexto y motivación

    ---

    <span style="background:#8c5a2b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">CONTEXTO</span>

    # Contexto histórico

    - **1973 · Black–Scholes–Merton.** Browniano geométrico: trayectorias continuas, retornos log-normales, volatilidad constante.
    - **Evidencia de discontinuidades.** Crash de octubre de 1987 (≈ −20 % en un día), anuncios de resultados, política monetaria, eventos geopolíticos.
    - **1976 · Merton**, *Option pricing when underlying stock returns are discontinuous* (JFE): difusión + Poisson compuesto de saltos multiplicativos.
    - **Después:** Bates (1996), Duffie–Pan–Singleton (2000), Kou (2002).

    > **Pregunta guía.** ¿Cómo construimos, implementamos y calibramos un modelo que capture saltos en acciones e índices, y qué precio pagamos en rigor (completitud, identificación)?

    ---

    <span style="background:#8c5a2b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">CONTEXTO</span>

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
    # § 1 · El modelo y su martingala

    ---

    <span style="background:#2f855a;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">DEFINICIÓN</span>

    # Definición 1 · Ingredientes y supuestos

    Sea $(\Omega,\mathcal F,(\mathcal F_t)_{t\ge0},\mathbb Q)$ un espacio filtrado con las condiciones usuales (filtración completa y continua por la derecha); $W$, $N$ y $S$ son $(\mathcal F_t)$-adaptados.

    > **(H1)** $W$ es un $\mathbb Q$-movimiento browniano estándar y $\sigma>0$.  
    > **(H2)** $N$ es un proceso de Poisson homogéneo de intensidad $\lambda>0$, con tiempos de salto $\tau_1<\tau_2<\cdots$.  
    > **(H3)** $(J_i)_{i\ge1}$ son i.i.d., $J_i=\ln Y_i\sim\mathcal N(\mu_J,\sigma_J^2)$, $\sigma_J>0$.  
    > **(H4)** $W$, $N$ y $(J_i)$ son mutuamente independientes.

    Los supuestos H1–H4 **no son teoremas**: son hipótesis del modelo. $\mathbb Q$ se toma como **exógena** (neutral al riesgo). Merton (1976) supone además riesgo de salto *no sistemático* (diversificable).

    ---

    <span style="background:#2f855a;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">DEFINICIÓN</span>

    # Definición 2 · Compensador

    > Se llama **compensador** de los saltos a la constante

    $$m:=\mathbb E[Y-1]=e^{\mu_J+\sigma_J^2/2}-1 .$$


    Es finito porque $J$ es normal: $\mathbb E[e^{J}]=e^{\mu_J+\sigma_J^2/2}<\infty$ (fgm de la normal evaluada en 1).

    Interpretación: $m$ es el salto *relativo* medio de $S$. Con $\mu_J<0$ y $\sigma_J$ moderada, $m<0$.

    ---

    <span style="background:#2f855a;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">DEFINICIÓN</span>

    # Definición 3 · Modelo de Merton

    > Un proceso positivo $(S_t)$ sigue el **modelo de saltos por difusión de Merton** bajo $\mathbb Q$ si

    $$\frac{dS_t}{S_{t^-}}=(r-\lambda m)\,dt+\sigma\,dW_t+(Y-1)\,dN_t,\qquad S_0>0,$$

    > con $r$ la tasa libre de riesgo constante y H1–H4.

    - $S_{t^-}=\lim_{s\uparrow t}S_s$: el integrando es **predecible**.
    - En un salto $\tau_i$: $S_{\tau_i}=S_{\tau_i^-}\,Y_i$, es decir, el cambio relativo es $Y_i-1$.
    - Interpretación en renta variable: anuncios de resultados, sorpresas de política monetaria, crisis geopolíticas, fusiones y quiebras.
    - **EDS** = ecuación diferencial estocástica; **MJD** = *Merton jump-diffusion*, el nombre en inglés de este modelo.

    ---

    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 1 · Un salto y el conteo de saltos

    **(a) Un salto.** Una acción cotiza en $S_{\tau^-}=100$ y ocurre un salto con $J=\mu_J=-0.10$. Entonces $Y=e^{-0.10}$ y

    $$S_\tau=100\,e^{-0.10}=90.48\quad(\text{caída de }9.52\,\%).$$

    **(b) Conteo.** Para $\lambda T=5$ (acción con resultados trimestrales, $T=1$):

    | $\mathbb P(N_T=0)$ | $\mathbb P(N_T=5)$ | $\mathbb P(N_T\ge1)$ |
    |---|---|---|
    | $e^{-5}=0.0067$ | $e^{-5}5^5/5!=0.1755$ | $0.9933$ |

    Con $\lambda T=1$: $\mathbb P(N_T=0,1,2)=0.3679,\ 0.3679,\ 0.1839$.

    ---

    <span style="background:#3b7ea1;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">LEMA</span>

    # Lema 1 · Los saltos compensados forman una martingala

    > Sea $M_t:=\sum_{i=1}^{N_t}(Y_i-1)-\lambda m\,t$. Entonces $M$ es una $\mathbb Q$-martingala.

    *Demostración.* $M$ tiene incrementos independientes y estacionarios (suma de un Poisson compuesto y una recta). Además, por la identidad de Wald (condicionando en $N_t$ e independencia de $N$ y $J$):

    $$\mathbb E\Big[\sum_{i=1}^{N_t}(Y_i-1)\Big]=\mathbb E[N_t]\,\mathbb E[Y-1]=\lambda t\,m .$$

    Luego $\mathbb E[M_t]=0$ y $\mathbb E[M_t-M_s\mid\mathcal F_s]=\mathbb E[M_{t-s}]=0$. Integrabilidad: $\mathbb E|Y-1|<\infty$. $\square$

    Consecuencia: $\dfrac{dS_t}{S_{t^-}}=r\,dt+\sigma\,dW_t+dM_t$.

    ---

    <span style="background:#0e7490;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">TEOREMA</span>

    # Teorema 1 · Solución explícita

    > Bajo la Definición 3,

    $$S_t=S_0\exp\!\Big[\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)t+\sigma W_t+\sum_{i=1}^{N_t}J_i\Big].$$


    *Demostración.* Entre saltos, $S$ es continua y la fórmula de Itô aplicada a $\ln S$ da $d\ln S=(r-\lambda m-\tfrac{\sigma^2}{2})dt+\sigma\,dW$. **En un tiempo de salto** $\tau_i$:

    $$\ln S_{\tau_i}-\ln S_{\tau_i^-}=\ln\frac{S_{\tau_i^-}Y_i}{S_{\tau_i^-}}=\ln Y_i=J_i .$$

    Sumando la parte continua y los saltos hasta $t$ se obtiene la expresión. La unicidad sigue de la unicidad de la solución de la EDS lineal con saltos. $\square$
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    <span style="background:#0e7490;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">TEOREMA</span>

    # Teorema 2 · Neutralidad al riesgo

    > $\big(e^{-rt}S_t\big)_{t\ge0}$ es una $\mathbb Q$-martingala. En particular $\mathbb E^{\mathbb Q}[S_t]=S_0e^{rt}$.

    *Demostración.* Por H4, $\mathbb E[S_t]$ se factoriza:

    $$\mathbb E[S_t]=S_0e^{(r-\lambda m)t}\underbrace{\mathbb E\big[e^{\sigma W_t-\sigma^2t/2}\big]}_{=1}\;\underbrace{\mathbb E\Big[\prod_{i=1}^{N_t}Y_i\Big]}_{=\,e^{\lambda m t}}.$$

    El último factor, condicionando en $N_t$: $\sum_{k\ge0}e^{-\lambda t}\frac{(\lambda t)^k}{k!}(1+m)^k=e^{\lambda t m}$. Por tanto $\mathbb E[S_t]=S_0e^{rt}$.

    La propiedad de martingala sale del Lema 1 y de que $e^{-rt}S_t=S_0\,\mathcal E(\sigma W+M)_t$ (exponencial de Doléans-Dade) es positiva e integrable. $\square$

    ---

    <span style="background:#c58b00;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">COROLARIO</span>

    # Corolarios 1 y 2

    > **Cor. 1 (esperanza condicional).** Para $s\le t$: $\ \mathbb E[S_t\mid\mathcal F_s]=S_s\,e^{r(t-s)}$.  
    > **Cor. 2 (precio de un forward).** El precio justo de un forward a plazo $T$ sobre $S$ es $F_0=S_0e^{rT}$, independiente de $\lambda,\mu_J,\sigma_J,\sigma$.

    *Demostración.* Cor. 1: el Teorema 2 y la propiedad de incrementos independientes dan $\mathbb E[S_t/S_s\mid\mathcal F_s]=\mathbb E[S_{t-s}]/S_0=e^{r(t-s)}$. Cor. 2: valor esperado descontado del payoff $S_T-F_0$ igual a cero. $\square$

    La independencia de los parámetros de salto es **crucial**: los saltos no cambian el precio del forward, solo la distribución.

    ---

    <span style="background:#3b7ea1;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">PROPOSICIÓN</span>

    # Proposición 1 · $\ln S$ es un proceso de Lévy

    > $X_t=\ln(S_t/S_0)$ es un proceso de Lévy con tripleta $(b,\sigma^2,\nu)$:

    $$b=r-\lambda m-\tfrac{\sigma^2}{2}\ \ (\text{sin truncar}),\qquad \nu(dx)=\lambda f_J(x)\,dx,\quad \nu(\mathbb R)=\lambda<\infty .$$


    *Demostración.* $X_t=bt+\sigma W_t+\sum_{i\le N_t}J_i$ es suma de un browniano con deriva y un Poisson compuesto independiente; ambos son Lévy y su suma (independientes) también. La medida de Lévy del Poisson compuesto es $\lambda\,\mathbb P(J\in dx)$. $\square$

    Es de **actividad finita**: casi seguramente hay un número finito de saltos en $[0,T]$. Contraste: VG, CGMY y NIG tienen actividad infinita.

    ---

    <span style="background:#b4372b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJERCICIO</span>

    # Ejercicio 1 · El compensador

    Considere la EDS **sin** compensador:

    $$\frac{dS_t}{S_{t^-}}=r\,dt+\sigma\,dW_t+(Y-1)\,dN_t .$$

    1. Calcule $\mathbb E[S_t]$ en función de $S_0,r,\lambda,m$.
    2. Evalúe numéricamente con $S_0=100$, $r=0.05$, $\lambda=5$, $\mu_J=-0.1$, $\sigma_J=0.2$, $t=1$.
    3. ¿Es $e^{-rt}S_t$ una martingala? ¿Qué error de valoración cometería?

    > *Pista:* repita el cálculo del Teorema 2 con la deriva $r$ en lugar de $r-\lambda m$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # § 2 · Ley y cumulantes

    ---

    <span style="background:#3b7ea1;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">PROPOSICIÓN</span>

    # Proposición 2 · Ley condicional y densidad

    > Dado $N_T=k$, el log-retorno $X=\ln(S_T/S_0)$ satisface

    $$X\mid N_T=k\sim\mathcal N(\mu_k,s_k^2),\quad \mu_k=\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)T+k\mu_J,\quad s_k^2=\sigma^2T+k\sigma_J^2 .$$

    > Por tanto la densidad es la **mezcla Poisson de normales**

    $$f_X(x)=\sum_{k\ge0}e^{-\lambda T}\frac{(\lambda T)^k}{k!}\,\varphi(x;\mu_k,s_k^2).$$


    *Demostración.* Dado $N_T=k$, $\sum_{i\le k}J_i\sim\mathcal N(k\mu_J,k\sigma_J^2)$ e independiente de $W_T$ (H4). La suma de normales independientes es normal. La densidad marginal se obtiene por la ley de probabilidad total. $\square$
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    <span style="background:#0e7490;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">TEOREMA</span>

    # Teorema 3 · Función generadora de cumulantes

    > Sea $\psi_X(u):=\ln\mathbb E[e^{uX}]=\sum_{n\ge1}\kappa_n\frac{u^n}{n!}$ (los $\kappa_n$ son los **cumulantes**). Para $u\in\mathbb R$,

    $$\psi_X(u)=u\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)T+\tfrac{\sigma^2u^2}{2}T+\lambda T\big(e^{u\mu_J+u^2\sigma_J^2/2}-1\big).$$

    > En consecuencia, con $\kappa_n=\lambda T\,\mathbb E[J^n]$ para $n\ge3$:

    $$\kappa_2=\sigma^2T+\lambda T(\sigma_J^2+\mu_J^2),\ \ \kappa_3=\lambda T(\mu_J^3+3\mu_J\sigma_J^2),\ \ \kappa_4=\lambda T(\mu_J^4+6\mu_J^2\sigma_J^2+3\sigma_J^4).$$


    *Demostración.* Por H4, $\mathbb E[e^{uX}]$ es el producto de las fgm del browniano y del Poisson compuesto. Para el Poisson compuesto, $\mathbb E[e^{u\sum J_i}]=\sum_k P_k\,\mathbb E[e^{uJ}]^k=\exp\{\lambda T(\mathbb E[e^{uJ}]-1)\}$ con $\mathbb E[e^{uJ}]=e^{u\mu_J+u^2\sigma_J^2/2}$. Los cumulantes del Poisson compuesto son $\lambda T\,\mathbb E[J^n]$. $\square$

    ---

    <span style="background:#c58b00;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">COROLARIO</span>

    # Corolarios 3 y 4 · Asimetría, curtosis y BS

    > **Cor. 3.** Sean $\gamma_1=\kappa_3/\kappa_2^{3/2}$ (asimetría) y $\gamma_2=\kappa_4/\kappa_2^{2}$ (exceso de curtosis). Entonces $\gamma_1=O(T^{-1/2})$, $\gamma_2=O(T^{-1})$; con $\mu_J<0$, $\gamma_1<0$ y $\gamma_2>0$.  
    > **Cor. 4.** Si $\lambda=0$, $X$ es normal ($\kappa_n=0$, $n\ge3$) y el modelo se reduce a Black–Scholes.

    *Demostración.* Cor. 3: $\kappa_2,\kappa_3,\kappa_4$ son **lineales en $T$**, luego $\gamma_1\propto T/T^{3/2}$ y $\gamma_2\propto T/T^{2}$. Si $\mu_J<0$, $\kappa_3=\lambda T\mu_J(\mu_J^2+3\sigma_J^2)<0$ y $\kappa_4>0$ (suma de términos no negativos). Cor. 4: sustituir $\lambda=0$ en el Teorema 3. $\square$

    La **normalidad se recupera** a plazos largos (teorema central del límite).
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Verificación numérica de los Corolarios 3 y 4:** el gráfico muestra $|\gamma_1|$ y $\gamma_2$
    contra $T$ en escala log-log, con las pendientes teóricas $-1/2$ y $-1$.
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
    _ax.set_title("Asimetría y exceso de curtosis de ln(S_T/S_0)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 2 · Momentos de una acción

    $S_0=100$, $r=0.05$, $\sigma=0.2$, $\lambda=5$, $\mu_J=-0.1$, $\sigma_J=0.2$, $T=1$:

    $\kappa_2=0.04+5(0.04+0.01)=0.29$

    $\kappa_3=5\,(-0.001-0.012)=-0.065$

    $\kappa_4=5\,(0.0001+0.0024+0.0048)=0.0365$

    | | teoría | simulado ($2\times10^6$) |
    |---|---|---|
    | $\gamma_1$ | −0.4162 | −0.4152 |
    | $\gamma_2$ | 0.4340 | 0.4290 |

    La normal con igual media y varianza no reproduce la cola izquierda.

    ---

    <span style="background:#b4372b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJERCICIO</span>

    # Ejercicio 2 · Escalamiento con el plazo

    Con los parámetros del Ejemplo 2:

    1. Calcule $\gamma_1$ y $\gamma_2$ para $T=0.25$ y $T=4$ **sin recalcular los cumulantes**, usando el Corolario 3.
    2. ¿Para qué plazo la aproximación normal es mejor? Justifique.
    3. Demuestre que, para $\mu_J=0$, $\gamma_1=0$ pero $\gamma_2>0$ si $\lambda>0$.

    > *Pista:* $\kappa_2,\kappa_3,\kappa_4$ son proporcionales a $T$, así que $\gamma_1(T)=\gamma_1(1)\,T^{-1/2}$ y $\gamma_2(T)=\gamma_2(1)\,T^{-1}$.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *La comprobación por simulación del Ejemplo 2 (con $2\times10^6$ trayectorias) está en la Hora 2,
    después de la Simulación 1 (**Laboratorio 7**).*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#0e7490;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">TEOREMA</span>

    # Teorema 4 · Función característica


    $$\chi_T(u)=\mathbb E[e^{iuX}]=\exp\!\Big[iu\big(r-\lambda m-\tfrac{\sigma^2}{2}\big)T-\tfrac{\sigma^2u^2}{2}T+\lambda T\big(e^{iu\mu_J-u^2\sigma_J^2/2}-1\big)\Big].$$


    *Demostración.* Idéntica al Teorema 3, sustituyendo $u\mapsto iu$ (la función característica de la normal es $e^{iu\mu-u^2\sigma^2/2}$ y la de un Poisson compuesto es $\exp\{\lambda T(\chi_J(u)-1)\}$ con $\chi_J(u)=\mathbb E[e^{iuJ}]=e^{iu\mu_J-u^2\sigma_J^2/2}$). $\square$

    Estructura: **difusión gaussiana** + **deriva compensada** + **Poisson compuesto**.

    Es la entrada de los métodos de Fourier: con un factor de amortiguamiento $\delta>0$ (que hace integrable el precio), Carr–Madan expresa la call como transformada inversa de $\chi_T$ evaluada en $u-i(\delta+1)$ (calibración sobre superficies; tarea de extensión).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # § 3 · Valoración de opciones europeas

    ---

    <span style="background:#3b7ea1;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">LEMA</span>

    # Lema 2 · Call con log-precio normal

    > Si $\ln S\sim\mathcal N(\mu,s^2)$, entonces

    $$\mathbb E[(S-K)^+]=e^{\mu+s^2/2}\,\Phi(d_1)-K\,\Phi(d_2),\quad d_1=\frac{\mu-\ln K+s^2}{s},\ \ d_2=d_1-s .$$


    *Demostración.* $\mathbb E[(S-K)^+]=\mathbb E[S\,\mathbf 1_{S>K}]-K\,\mathbb P(S>K)$. Segundo término: $\mathbb P(\ln S>\ln K)=\Phi(d_2)$ con $d_2=(\mu-\ln K)/s$. Primer término: el cambio de medida con densidad $S/\mathbb E[S]$ desplaza la media del log a $\mu+s^2$ y da $e^{\mu+s^2/2}\,\Phi(d_1)$. $\square$

    Es la fórmula de Black–Scholes escrita para una normal general (sin referencia a $T$ ni $r$).

    ---

    <span style="background:#0e7490;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">TEOREMA</span>

    # Teorema 5 · Fórmula de Merton

    > Sea $P_k=e^{-\lambda T}\frac{(\lambda T)^k}{k!}$, $\ r_k=r-\lambda m+\frac kT\big(\mu_J+\tfrac{\sigma_J^2}{2}\big)$, $\ \sigma_k^2=\sigma^2+\frac{k\sigma_J^2}{T}$. El precio de la call europea es

    $$C_0=\sum_{k\ge0}P_k\Big[S_0e^{(r_k-r)T}\Phi(d_1^k)-Ke^{-rT}\Phi(d_2^k)\Big],\ \ d_1^k=\tfrac{\ln(S_0/K)+(r_k+\sigma_k^2/2)T}{\sigma_k\sqrt T},\ d_2^k=d_1^k-\sigma_k\sqrt T .$$


    *Demostración.* (1) Por H4 y el Teorema 2, $C_0=e^{-rT}\sum_kP_k\,\mathbb E[(S_T-K)^+\mid N_T=k]$. (2) Dado $k$, $\ln S_T\sim\mathcal N(\ln S_0+\mu_k,s_k^2)$ (Prop. 2) y $e^{\mu_k+s_k^2/2}=e^{(r-\lambda m)T+k(\mu_J+\sigma_J^2/2)}=e^{r_kT}$. (3) Aplicar el Lema 2 con $s_k^2=\sigma_k^2T$ y multiplicar por $e^{-rT}$. $\square$
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Del papel al código.** `merton_terms` evalúa $\sum_kP_k'\,\mathrm{BS}(r_k,\sigma_k)$ **vectorizado**
    (sobre $k$ y sobre $S_0$); `merton_breakdown` devuelve los términos uno a uno para poder inspeccionarlos;
    `merton_iv` invierte Black–Scholes (Definición 4).
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
def _(mo):
    mo.md(r"""
    <span style="background:#c58b00;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">COROLARIO</span>

    # Corolario 5 · Forma de mezcla Poisson de BS

    > Sea $\lambda':=\lambda(1+m)$ y $P_k':=e^{-\lambda'T}\frac{(\lambda'T)^k}{k!}$. Entonces

    $$C_0=\sum_{k\ge0}P_k'\;\mathrm{BS}(S_0,K,T,r_k,\sigma_k),$$

    > donde $\mathrm{BS}(S_0,K,T,\rho,s)$ es el precio de Black–Scholes de una call con tasa $\rho$ y volatilidad $s$ (Notación 3/3).

    *Demostración.* $\mathrm{BS}(\cdot,r_k,\sigma_k)=S_0\Phi(d_1^k)-Ke^{-r_kT}\Phi(d_2^k)$, luego $P_ke^{(r_k-r)T}\,\mathrm{BS}$ reproduce el término del Teorema 5. Además, como $e^{\mu_J+\sigma_J^2/2}=1+m$,

    $$P_ke^{(r_k-r)T}=e^{-\lambda T}\tfrac{(\lambda T)^k}{k!}e^{-\lambda mT}(1+m)^k=P_k'.\;\square$$

    > **Atención:** *no* es $\sum_kP_k\,\mathrm{BS}$ con intensidad $\lambda$; hay que usar $\lambda'$ (o el factor $e^{(r_k-r)T}$). Es el error más frecuente en la literatura.

    ---

    <span style="background:#c58b00;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">COROLARIO</span>

    # Corolarios 6 y 7 · Casos límite y paridad

    > **Cor. 6.** (i) Si $\lambda=0$: $P_0=1$, $r_0=r$, $\sigma_0=\sigma$ y $C_0=\mathrm{BS}(S_0,K,T,r,\sigma)$. (ii) Si $K\to0$: $C_0\to S_0$.  
    > **Cor. 7 (paridad put–call).** $\ \mathrm{Put}_0=C_0-S_0+Ke^{-rT}$.

    *Demostración.* Cor. 6 (i): sustitución directa. (ii): $d_1^k,d_2^k\to+\infty$, así que $\Phi\to1$ y $C_0\to S_0\sum_kP_k'=S_0$. Cor. 7: $(S_T-K)^+-(K-S_T)^+=S_T-K$; tomar $\mathbb E^{\mathbb Q}$ y descontar, usando el Teorema 2 ($\mathbb E[e^{-rT}S_T]=S_0$). $\square$

    La paridad **no depende** de los saltos: solo de la propiedad de martingala.
    """)
    return



@app.cell(hide_code=True)
def _(bs_call, math, merton_call, merton_put, mo):
    # Parámetros del Ejemplo 3 (= Actividad en parejas)
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
    **Comprobación numérica de los Corolarios 6 y 7** ($S_0=K=100$, $r=0.05$, $\sigma=0.2$, $T=1$, $\lambda=1$, $\mu_J=-0.2$, $\sigma_J=0.3$)

    | Propiedad | Resultado |
    |---|---|
    | Cor. 6 (i), $\lambda=0$: $C_0^{{\text{{Merton}}}}-C^{{\text{{BS}}}}$ | {_bs_diff:.2e} |
    | Cor. 6 (ii), $K\to0$: $C_0/S_0$ | {_lim:.8f} |
    | Cor. 7, paridad: $\mathrm{{Put}}_0$ (suma) − ($C_0-S_0+Ke^{{-rT}}$) | {_parity:.2e} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 3 · La fórmula término a término

    $S_0=K=100$, $T=1$, $r=0.05$, $\sigma=0.2$, $\lambda=1$, $\mu_J=-0.2$, $\sigma_J=0.3$.  
    $m=e^{-0.155}-1=-0.1436$, $\ \lambda'=0.8564$.

    | $k$ | $r_k$ | $\sigma_k$ | $P_k'$ | $\mathrm{BS}_k$ | $P_k'\,\mathrm{BS}_k$ | suma |
    |---|---|---|---|---|---|---|
    | 0 | 0.1936 | 0.2000 | 0.4247 | 19.2022 | 8.1548 | 8.1548 |
    | 1 | 0.0386 | 0.3606 | 0.3637 | 16.0077 | 5.8221 | 13.9769 |
    | 2 | −0.1164 | 0.4690 | 0.1557 | 14.1049 | 2.1967 | 16.1736 |
    | 3 | −0.2714 | 0.5568 | 0.0445 | 12.6125 | 0.5607 | 16.7344 |
    | 4 | −0.4264 | 0.6325 | 0.0095 | 11.3596 | 0.1081 | 16.8425 |
    | 5 | −0.5814 | 0.7000 | 0.0016 | 10.2774 | 0.0168 | 16.8592 |

    La suma converge a $C_0=16.8617$ (frente a BS con $\sigma=0.2$: 10.4506).
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 3 · ¿Cuánto aporta cada término de la suma? (Ejemplo 3)

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
    <span style="background:#3b7ea1;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">PROPOSICIÓN</span>

    # Proposición 3 · Cota del error de truncamiento

    > Sea $C^{(K^*)}$ la suma de Merton truncada en $k\le K^*$ y $N'\sim\text{Poisson}(\lambda'T)$. Entonces

    $$0\le C_0-C^{(K^*)}\le S_0\,\mathbb P(N'>K^*).$$


    *Demostración.* Por el Corolario 5, $C_0-C^{(K^*)}=\sum_{k>K^*}P_k'\,\mathrm{BS}_k$. Cada $\mathrm{BS}_k\ge0$ (es un precio de call) y $\mathrm{BS}_k\le S_0$ (la call no vale más que el subyacente). Luego el resto está entre $0$ y $S_0\sum_{k>K^*}P_k'=S_0\,\mathbb P(N'>K^*)$. $\square$

    > Es una cota **explícita y verificable a priori**: no depende de $K$ ni de $\sigma$.

    ---

    <span style="background:#c58b00;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">COROLARIO</span>

    # Corolario 8 · Elección de $K^*$

    > Para $K^*>\lambda'T$ (Chernoff): $\ \mathbb P(N'>K^*)\le e^{-\lambda'T}\Big(\dfrac{e\,\lambda'T}{K^*}\Big)^{K^*}$.  
    > Por tanto, dado $\varepsilon>0$, basta elegir el menor $K^*$ con $S_0\,\mathbb P(N'>K^*)<\varepsilon$ para garantizar error de precio menor que $\varepsilon$.

    *Demostración.* Para $\xi>0$, $\mathbb P(N'\ge K^*)\le e^{-\xi K^*}\mathbb E[e^{\xi N'}]=\exp\{-\xi K^*+\lambda'T(e^\xi-1)\}$. Minimizando en $\xi=\ln(K^*/\lambda'T)$ resulta la cota. $\square$

    Regla práctica: $\lambda'T\approx1$ ⇒ $K^*=15$ sobra; $K^*=30$ es conservador.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Verificación numérica del Teorema 4.** La tabla compara la suma de Merton (Teorema 5) con la
    inversión de Fourier de Carr–Madan a partir de $\chi_T$ (parámetros del Ejemplo 3). `merton_cf` implementa $\chi_T$.
    """)
    return


@app.cell
def _(comp_m, math, np, simpson):
    def merton_cf(u, T, r, sigma, lam, muJ, sigJ):
        """Función característica χ_T(u) de ln(S_T/S_0). Admite u complejo."""
        m = comp_m(muJ, sigJ)
        return np.exp(
            1j * u * (r - lam * m - 0.5 * sigma**2) * T
            - 0.5 * sigma**2 * u**2 * T
            + lam * T * (np.exp(1j * u * muJ - 0.5 * u**2 * sigJ**2) - 1)
        )

    def carr_madan_call(S0, K, T, r, sigma, lam, muJ, sigJ, alpha=1.5, vmax=200.0, npts=40001):
        """Carr–Madan; `alpha` es el factor de amortiguamiento (δ en las diapositivas)."""
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
    f_alpha = mo.ui.slider(0.5, 3.0, step=0.25, value=1.5, label="δ (amortiguación)")
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
    # § 4 · Volatilidad implícita

    ---

    <span style="background:#2f855a;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">DEFINICIÓN</span>

    # Definición 4 · Volatilidad implícita

    > Dado el precio $C^{\mathrm{mod}}(K,T)$ de una call, su **volatilidad implícita (IV)** $\Sigma(K,T)$ es la única solución de

    $$\mathrm{BS}(S_0,K,T,r,\Sigma)=C^{\mathrm{mod}}(K,T).$$


    Existe y es única porque $\Sigma\mapsto\mathrm{BS}(\cdot,\Sigma)$ es estrictamente creciente (vega $>0$) entre las cotas de no arbitraje $\max(S_0-Ke^{-rT},0)<C^{\mathrm{mod}}<S_0$. Numéricamente (`bs_call` es $\mathrm{BS}$ con $\rho=r$; `brentq` halla la raíz por el método de Brent):

    ```python
    brentq(lambda s: bs_call(S0, K, T, r, s) - price, 1e-4, 5.0)
    ```

    El **smile** (o **skew**) es la gráfica de $K\mapsto\Sigma(K,T)$. Bajo BS es una recta horizontal.

    ---

    <span style="background:#3b7ea1;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">PROPOSICIÓN</span>

    # Proposición 4 · Aplanamiento del smile con el plazo

    > Para $\mu_J\ne0$ fija, la asimetría $\gamma_1$ y la curtosis $\gamma_2$ de $X$ tienden a cero a ritmos $T^{-1/2}$ y $T^{-1}$ (Cor. 3). Heurísticamente (TCL), el smile **se aplana** y la IV ATM tiende a

    $$\Sigma_\infty=\sqrt{\kappa_2/T}=\sqrt{\sigma^2+\lambda(\sigma_J^2+\mu_J^2)} .$$


    *Esquema.* El Corolario 3 da la convergencia de los momentos estandarizados; por el TCL, $X$ se vuelve normal con varianza total $\kappa_2$ y la IV de cualquier strike converge a $\sqrt{\kappa_2/T}$ (resultado heurístico: no se demuestra aquí). $\square$

    > **Observación.** A plazo corto el precio OTM es $\approx\lambda T\,\mathbb E[(S_0Y-K)^+]$, lineal en $T$, y la IV OTM crece sin cota cuando $T\to0$. **Límite:** el skew empírico decae más despacio que $T^{-1/2}$; el MJD no lo reproduce.

    ---

    <span style="background:#6b7f6e;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">OBSERVACIÓN</span>

    # Limitaciones estructurales

    1. **Mercado incompleto.** Dos fuentes de riesgo ($W$, $N$) y un solo activo: no hay cartera replicante; hay infinitas medidas martingala equivalentes. Se necesita un criterio para elegir $\mathbb Q$ (varianza mínima, entropía mínima, Esscher, utilidad exponencial).
    2. **Actividad finita.** $\nu(\mathbb R)=\lambda<\infty$: no captura actividad infinita (VG, CGMY, NIG; ver Glosario 1/4).
    3. **Saltos lognormales i.i.d.** Asimetría con $\mu_J<0$, pero no la estructura temporal del skew.
    4. **Identificación.** Con $\lambda\Delta$ pequeño (datos diarios) los saltos raros se confunden con ruido difusivo: **identificación débil** (se cuantifica en la Hora 2).

    ---

    <span style="background:#6b7f6e;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">OBSERVACIÓN</span>

    # Extensiones y transición

    **Para conocer, no desarrollar hoy:**

    - **Kou (2002):** saltos exponenciales dobles ⇒ fórmula cerrada, mejor skew en acciones.
    - **Bates (1996):** Heston + saltos ⇒ estándar de industria en índices.
    - **Duffie–Pan–Singleton (2000):** marco afín con función característica cerrada.

    > **Transición.** El modelo es incompleto y sus parámetros no son observables. ¿Cómo lo simulamos *exactamente*, cómo verificamos el precio y cómo lo calibramos con datos de acciones e índices?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **HORA 2**

    # Simulación y aplicación

    ---

    <span style="background:#c2570c;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">SIMULACIÓN</span>

    # Simulación exacta de la ley terminal

    ```python
    def simulate_merton(S0, r, sigma, lam, muJ, sigmaJ, T, n, n_paths, seed=0):
        rng = np.random.default_rng(seed); dt = T / n
        m = np.exp(muJ + 0.5 * sigmaJ**2) - 1
        dN = rng.poisson(lam * dt, size=(n_paths, n))
        J = rng.normal(muJ * dN, sigmaJ * np.sqrt(np.maximum(dN, 1))) * (dN > 0)
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n))
        log_inc = (r - lam * m - 0.5 * sigma**2) * dt + sigma * dW + J
        log_S = np.log(S0) + np.cumsum(log_inc, axis=1)
        return np.concatenate([np.full((n_paths, 1), S0), np.exp(log_S)], axis=1), dN
    ```

    - **Exacta, no Euler:** incrementos de $\ln S$ independientes y estacionarios con ley conocida; dado $N_\Delta=k$, la suma de saltos es $\mathcal N(k\mu_J,k\sigma_J^2)$. Sin error de discretización en la **ley** de $S_{t_j}$.
    - **Si $dN=2$:** la suma es $\mathcal N(2\mu_J,2\sigma_J^2)$, **no** $2\cdot\mathcal N(\mu_J,\sigma_J^2)$.
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
    *Arriba, `simulate_merton` simula **trayectorias completas** (rejilla de $n$ pasos) y
    `simulate_terminal` simula solo $S_T$ **en un paso** (basta para precios europeos).
    El argumento `compensate=False` permite quitar $-\lambda m$ para romper la martingala a propósito.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#c2570c;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">SIMULACIÓN</span>

    # Simulación 1 · Trayectorias de una acción

    $S_0=100$, $r=0.05$, $\sigma=0.2$, $\lambda=5$, $\mu_J=-0.1$, $\sigma_J=0.2$, $T=1$ (acción con resultados trimestrales).

    **Chequeos, $2\times10^6$ trayectorias:**

    | | simulado | teoría |
    |---|---|---|
    | $\mathbb E[S_T]$ | 105.090 ± 0.039 | 105.127 |
    | $\mathbb E[N_T]$ | 5.0007 | 5 |
    | $\mathrm{Var}(N_T)$ | 4.9975 | 5 |

    Los $\pm$ son errores estándar. $z=-0.94$ (Notación 3/3): la martingala no se rechaza; media ≈ varianza ⇒ Poisson.
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

    Usa los **mismos parámetros y la misma semilla** de la Simulación 1. Izquierda: media móvil de $S_T$
    con compensador (negro) y **sin** compensador (rojo); las líneas punteadas son las teorías
    $S_0e^{rT}$ y $S_0e^{(r+\lambda m)T}$ (Ejercicio 1). Derecha: histograma de $N_T$ contra la pmf de Poisson.
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
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 2 · verificación por simulación (Hora 1 → Hora 2)

    Mismos parámetros del Ejemplo 2, $T=1$: $\kappa_2=0.29$, $\gamma_1=-0.4162$ (sim. $-0.4152$), $\gamma_2=0.4340$ (sim. $0.4290$).
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

    Misma configuración del Ejemplo 2. Estandarizamos $\ln(S_T/S_0)$ y contamos la frecuencia de
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
    <span style="background:#c2570c;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">SIMULACIÓN</span>

    # Suma truncada de Merton

    ```python
    def merton_call(S0, K, T, r, sigma, lam, muJ, sigmaJ, Ktrunc=30):
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

    > **Actividad en parejas (8 min).** $S_0=K=100$, $r=0.05$, $\sigma=0.2$, $T=1$, $\lambda=1$, $\mu_J=-0.2$, $\sigma_J=0.3$. Compara con Monte Carlo, varía $K^*\in\{5,10,15,20,30\}$ y justifica $K^*$ con la cota de Poisson.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#c2570c;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">SIMULACIÓN</span>

    # Simulación 2 · Suma vs. Monte Carlo

    | Método | Call |
    |---|---|
    | **Merton ($K^*=30$)** | **16.8617** |
    | Monte Carlo ($2\times10^6$) | 16.8440 ± 0.0180 |
    | Black–Scholes ($\sigma=0.2$) | 10.4506 |
    | Put (paridad) | 11.9846 |

    Parámetros del Ejemplo 3. $z=-0.98$: coinciden dentro del error estándar.

    Los saltos negativos grandes **elevan el precio** frente a BS: $\kappa_2=0.04+1\cdot(0.09+0.04)=0.17$ vs. $0.04$.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *En el cuaderno la suma está vectorizada (`merton_terms`); la versión con bucle de la diapositiva
    anterior hace exactamente lo mismo, término a término. Los parámetros por defecto de abajo son los del
    Ejemplo 3 y de la actividad en parejas.*
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
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 4 · Error de truncamiento y cota

    Parámetros del Ejemplo 3 ($\lambda'T=0.856$, $S_0=100$). Error real $=\lvert C_0-C^{(K^*)}\rvert$, con $C_0$ calculado con $K^*=60$ (la suma vs. Monte Carlo está en la Simulación 2).

    | $K^*$ | error real | cota (Prop. 3) |
    |---|---|---|
    | 5 | 2.4·10⁻³ | 2.6·10⁻² |
    | 10 | 1.2·10⁻⁸ | 2.1·10⁻⁷ |
    | 15 | ≈ 10⁻¹⁴ | 1.8·10⁻¹³ |
    | 20 | < 10⁻¹⁵ | 3.3·10⁻²⁰ |
    | 30 | < 10⁻¹⁵ | 4.3·10⁻³⁵ |

    La cota **siempre** domina; bajo $10^{-14}$ el error observado es ruido de punto flotante. Con $K^*=10$ el error ya es $<10^{-6}$; $K^*=15$ sobra y $K^*=30$ es conservador (Cor. 8).
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
    _ax.semilogy(_ks, np.maximum(_bound, 1e-40), "s--", label=r"cota $S_0\,P(N'>K^*)$ (Prop. 3)")
    _ax.semilogy(_ks, _chern, "^:", label="cota de Chernoff (Cor. 8)")
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

    | K* | error real | cota (Prop. 3) |
    |---|---|---|
    {_rows}
    """
    )
    mo.hstack([_fig, _tbl], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#b4372b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJERCICIO</span>

    # Ejercicios 3 a 5

    **Ejercicio 3 (derivación).** Demuestre que $\mathbb E[S_T\mid N_T=k]=S_0e^{r_kT}$, es decir, que $e^{\mu_k+s_k^2/2}=e^{r_kT}$.

    **Ejercicio 4 (truncamiento).** Con los datos del Ejemplo 3 ($\lambda'T=0.856$, $S_0=100$):
    1. Halle el menor $K^*$ garantizando error $<10^{-6}$ y error $<10^{-10}$ usando la Proposición 3.
    2. Dado $C_0=16.8617$, calcule el precio del put con la paridad (Cor. 7).

    **Ejercicio 5 (computacional).** Implemente `merton_call` y compare con Monte Carlo ($10^5$ trayectorias) para $K\in\{80,100,120\}$. ¿Dónde el error de MC es mayor en términos relativos?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 5 · Smile de Merton

    $S_0=100$, $r=0.03$, $\sigma=0.15$, $\lambda=0.6$, $\mu_J=-0.15$, $\sigma_J=0.12$. IV (%) en $\ln(K/S_0)=\mp0.21$:

    | $T$ | put OTM | ATM | call OTM |
    |---|---|---|---|
    | 0.1 | 35.8 | 17.9 | 20.0 |
    | 0.5 | 24.4 | 19.7 | 17.5 |
    | 1.0 | 22.5 | 20.3 | 18.5 |
    | 2.0 | 21.6 | 20.5 | 19.5 |

    $\Sigma_\infty=\sqrt{0.15^2+0.6(0.12^2+0.15^2)}=21.1\,\%$.

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

    IV ATM a $T=30$: {_atm_long:.1f} % · límite heurístico $\Sigma_\infty=\sqrt{{\sigma^2+\lambda(\sigma_J^2+\mu_J^2)}}$: {_lim:.1f} %
    """
    )
    mo.hstack([_fig, _tbl], widths=[3, 2], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laboratorio 11 · Estructura temporal del skew (Proposición 4)

    Con los parámetros del Ejemplo 5 (arriba), medimos el **skew** $\text{IV}(\ln K/S_0=-0.2)-\text{IV}(+0.2)$
    para muchos plazos. La teoría dice que decae como $T^{-1/2}$ a plazos largos (línea punteada);
    el dato empírico decae **más lento**: es el límite del modelo.
    A la derecha, la IV ATM converge a $\Sigma_\infty=\sqrt{\sigma^2+\lambda(\sigma_J^2+\mu_J^2)}$.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#b4372b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJERCICIO</span>

    # Ejercicio 6 · Interpretar el smile

    Con la tabla del Ejemplo 5:

    1. Calcule el **skew** $\Sigma(\text{put OTM})-\Sigma(\text{call OTM})$ para $T=0.1$ y $T=2$. ¿Cuántas veces se reduce?
    2. ¿Qué parámetro del modelo explica que la IV de la put OTM sea mayor que la de la call OTM?
    3. Comente: ¿qué parte del smile empírico de SPX (skew persistente a plazos largos) **no** reproduce el modelo y por qué?

    > *Pista:* signo de $\mu_J$ y Proposición 4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # § 5 · Calibración por máxima verosimilitud

    ---

    <span style="background:#2f855a;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">DEFINICIÓN</span>

    # Definición 5 · Log-verosimilitud y MLE

    > Dados retornos logarítmicos $x_i=\ln(S_{t_i}/S_{t_{i-1}})$ i.i.d. con paso $\Delta=t_i-t_{i-1}$ (en años) y $\theta=(\sigma,\lambda,\mu_J,\sigma_J)$:

    $$\ell(\theta)=\sum_{i=1}^n\ln\sum_{k\ge0}\frac{e^{-\lambda\Delta}(\lambda\Delta)^k}{k!}\,\varphi\big(x_i;\mu_k^{(\Delta)},s_k^{2,(\Delta)}\big),$$


    $$\mu_k^{(\Delta)}=\big(\alpha-\lambda m-\tfrac{\sigma^2}{2}\big)\Delta+k\mu_J,\qquad s_k^{2,(\Delta)}=\sigma^2\Delta+k\sigma_J^2 .$$

    > El **estimador de máxima verosimilitud (MLE)** es $\hat\theta=\arg\max_{\theta\in\Theta}\ell(\theta)$.

    Los datos están bajo la medida **física** $\mathbb P$: la deriva es $\alpha$ (aquí $\alpha=r$ conocida; en datos reales es un parámetro más, débilmente identificado). Se evalúa con `logsumexp` truncando en $k\le k_{\max}$.

    ---

    <span style="background:#3b7ea1;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">PROPOSICIÓN</span>

    # Proposición 5 · Acotación de la verosimilitud

    > Sea $\Theta=\{\sigma\ge\sigma_{\min},\ \sigma_J\ge\sigma_{J,\min},\ \lambda\in[\lambda_{\min},\lambda_{\max}]\}$ con $\sigma_{\min}>0$. Entonces para todo $x$ y $\theta\in\Theta$,

    $$f_\theta(x)\le\frac{1}{\sqrt{2\pi\,\sigma_{\min}^2\Delta}},\qquad\text{luego}\ \ \ell(\theta)\le n\ln\frac{1}{\sqrt{2\pi\,\sigma_{\min}^2\Delta}}<\infty .$$


    *Demostración.* Cada componente normal tiene densidad $\le(2\pi s_k^{2,(\Delta)})^{-1/2}\le(2\pi\sigma^2\Delta)^{-1/2}$, y los pesos de Poisson suman 1: $f_\theta$ es una combinación convexa. $\square$

    Sin la cota $\sigma\ge\sigma_{\min}$ la verosimilitud puede **no estar acotada** (si $\sigma\to0$ y un dato cae sobre la media del componente $k=0$). Por eso se usa L-BFGS-B con cotas.

    ---

    <span style="background:#0e7490;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">TEOREMA</span>

    # Teorema 6 · MLE bajo mala especificación

    > Sea $g$ la densidad verdadera (posiblemente fuera de la familia). Bajo condiciones de regularidad (parámetro en compacto, continuidad, dominación integrable, minimizador único), $\hat\theta_n\xrightarrow{\text{c.s.}}\theta^*=\arg\min_\theta\mathrm{KL}(g\,\|\,f_\theta)$ y

    $$\sqrt n\,(\hat\theta_n-\theta^*)\xrightarrow{d}\mathcal N\big(0,\,A^{-1}BA^{-1}\big),$$

    > con $A=-\mathbb E[\nabla^2\ln f_{\theta^*}]$ y $B=\mathbb E[\nabla\ln f_{\theta^*}\nabla\ln f_{\theta^*}^\top]$ (White, 1982). Si el modelo es correcto, $A=B$ y la covarianza es $A^{-1}$.

    Resultado estándar; **no se demuestra** aquí.

    Lectura: la MLE estima el miembro de la familia **más cercano en divergencia KL**, no "el verdadero" modelo. Aquí $\mathrm{KL}(g\,\|\,f)=\int g\ln(g/f)\ge0$ (divergencia de Kullback–Leibler), con igualdad si y solo si $g=f$.

    ---

    <span style="background:#c58b00;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">COROLARIO</span>

    # Corolario 9 · Errores estándar

    > Si el modelo es correcto, $\widehat{\mathrm{se}}(\hat\theta_j)=\sqrt{\big[(-\nabla^2\ell(\hat\theta))^{-1}\big]_{jj}}$. Un número de condición muy grande de $-\nabla^2\ell(\hat\theta)$ señala **identificación débil**.

    *Demostración.* Con $A=B$ en el Teorema 6, $\mathrm{Cov}(\hat\theta)\approx\frac1n A^{-1}\approx(-\nabla^2\ell(\hat\theta))^{-1}$ (información observada). Si el menor autovalor es $\approx0$, el inverso tiene entradas grandes. $\square$

    La **verosimilitud perfilada** $\ell_p(\lambda)=\max_{\sigma,\mu_J,\sigma_J}\ell(\theta)$ es la herramienta más honesta: su intervalo al 95 % es $\{\lambda:\ell_p(\hat\lambda)-\ell_p(\lambda)\le1.92\}$.

    ---

    <span style="background:#c2570c;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">SIMULACIÓN</span>

    # Calibración por MLE · en la práctica

    *Calibrar es el problema inverso:* dados $x_1,\dots,x_n$ con paso $\Delta$, hallar $\theta=(\sigma,\lambda,\mu_J,\sigma_J)$ (Def. 5).

    - **Medida física.** Los datos viven bajo $\mathbb P$: deriva $\alpha$ (aquí $\alpha=r$ conocida, por simplicidad); en datos reales $\alpha$ es un parámetro más, muy débilmente identificado.
    - **Degeneración.** Con varianza libre la verosimilitud no está acotada si $\sigma\to0$: se impone $\sigma\ge\sigma_{\min}>0$ (L-BFGS-B con cotas; Prop. 5).
    - **Numérica.** `logsumexp` sobre $k=0,\dots,k_{\max}$; con $\lambda\Delta\approx0.02$, $k_{\max}=20$ basta.
    - **Modelo incorrecto:** la MLE converge al miembro de la familia más cercano en divergencia **KL** (Teo. 6).
    - **Datos sintéticos.** Permiten **conocer la verdad**; los alumnos repiten el código con `retornos_diarios_sp500.csv` o `retornos_diarios_aapl.csv` (Ejemplo 6, Ejercicio 8).
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Del papel al código.** `nll` es $-\ell(\theta)$: una matriz $n\times(k_{\max}+1)$ de log-densidades
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

    def nll(theta, x, dt=DT, alpha=ALPHA, kmax=20):
        """Menos log-verosimilitud de retornos x con paso dt (mezcla Poisson de normales)."""
        sigma, lam, muJ, sigJ = theta
        m = comp_m(muJ, sigJ)
        k = np.arange(kmax + 1)
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
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 6 · Calibración sobre datos sintéticos

    6048 retornos diarios (24 años, $\Delta=1/252$) con $\theta_0=(0.18,\,5,\,-0.03,\,0.04)$, $\alpha=r=0.02$, 5 puntos iniciales:

    | Parámetro | verdadero | MLE | error est. (Cor. 9) |
    |---|---|---|---|
    | $\sigma$ | 0.180 | 0.1814 | 0.0018 |
    | $\lambda$ | 5.00 | 4.147 | 0.687 |
    | $\mu_J$ | −0.030 | −0.0412 | 0.0063 |
    | $\sigma_J$ | 0.040 | 0.0405 | 0.0036 |

    - Asimetría muestral −1.69; exceso de curtosis 12.83.
    - 4 de 5 inicios llegan al mismo óptimo ($-\ell=-18080.530$); el inicio en $\theta_0$ se detiene 0.66 por encima: valle **plano**. Número de condición de la Hessiana $\approx1.6\times10^5$.
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

    | Parámetro | verdadero | MLE | error est. (Cor. 9) | error relativo |
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
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 7 · Perfil de verosimilitud de $\lambda$

    Con los datos del Ejemplo 6 se fija cada $\lambda$ y se reoptimizan $(\sigma,\mu_J,\sigma_J)$.

    - La curva es muy plana: el IC al 95 % de $\lambda$ abarca ≈ $[3,\,5.7]$ (semiancho relativo ≈ 33 %, o sea 1.96 × el 17 % del Ejemplo 6).
    - $\sigma$ y $\sigma_J$ se estiman con error relativo < 10 %.
    - Causa: con $\lambda\Delta\approx0.02$, solo ~2 % de los días contienen un salto, y un salto pequeño se parece a un día de alta volatilidad.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Requiere haber corrido el Ejemplo 6.*
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
    La estrella es la MLE y la cruz roja la verdad. *(Requiere el Ejemplo 6.)*
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

    Simulamos **muchas muestras** con los parámetros verdaderos del Ejemplo 6, calibramos cada una
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
    Usa los parámetros verdaderos y la semilla del Ejemplo 6.
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
    <span style="background:#b4372b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJERCICIO</span>

    # Ejercicios 7 y 8

    **Ejercicio 7 (conteo).** En el Ejemplo 6, $\lambda\Delta=5/252$.
    1. ¿Cuántos días con al menos un salto se esperan en 6048 retornos?
    2. ¿Por qué esto explica el error estándar relativo de $\hat\lambda$ (≈ 17 %) frente al de $\hat\sigma$ (≈ 1 %)?

    **Ejercicio 8 (computacional, en parejas).** Cargue `retornos_diarios_sp500.csv` (o `retornos_diarios_aapl.csv`; formato en Notación 3/3).
    1. Estime $\hat\theta$ con `minimize(..., method='L-BFGS-B')` desde dos puntos iniciales distintos.
    2. Reporte errores estándar (Hessiana numérica) y compare índice vs. acción individual.
    3. Discuta la estabilidad y la identificación débil.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Calibra con tus propios datos (Ejercicio 8).** Sube un CSV con retornos diarios o precios
    (p. ej. `retornos_diarios_sp500.csv`, `retornos_diarios_aapl.csv`; formato en Notación 3/3).
    Se usa la columna `ret`/`return`/`close`/`price` si existe; si no, la última numérica.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # § 6 · Cobertura e incompletitud

    ---

    <span style="background:#2f855a;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">DEFINICIÓN</span>

    # Definición 6 · Replicación y completitud

    > Un **portafolio autofinanciado** en $S$ y la cuenta bancaria es $V_t=\Delta_tS_t+\beta_te^{rt}$ con $\Delta$ predecible y $dV=\Delta\,dS+r(V-\Delta S)\,dt$. Una opción con payoff $G$ es **replicable** si existe $(\Delta,\beta)$ con $V_T=G$ c.s. El mercado es **completo** si todo payoff acotado es replicable.

    Black–Scholes es completo: un activo, **una** fuente de riesgo. Merton tiene **dos** fuentes de riesgo ($W$ y $N$) y **un** activo.

    ---

    <span style="background:#0e7490;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">TEOREMA</span>

    # Teorema 7 · Incompletitud del mercado

    > Sea $C(t,S)$ el precio de la call en $t$ (Teorema 5 con $(S_0,T)$ sustituidos por $(S,T-t)$), estrictamente convexo en $S$ para $t<T$. Bajo el modelo de Merton **no existe** una estrategia autofinanciada que replique la call.

    *Esquema de la demostración.* Si $V_t=C(t,S_t)$, la fórmula de Itô con saltos exige igualar las dos partes de $dV=dC$:
    - **parte difusiva:** $\Delta_t=\partial_SC(t,S_{t^-})$;
    - **parte de salto:** $\Delta_tS_{t^-}(y-1)=C(t,S_{t^-}y)-C(t,S_{t^-})$ **para todo** $y$ en el soporte de $Y$, que es $(0,\infty)$.

    La segunda condición dice que $y\mapsto C(t,S\,y)$ es **lineal**, lo que contradice la convexidad estricta de $C$. $\square$

    ---

    <span style="background:#c58b00;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">COROLARIO</span>

    # Corolario 10 · El delta-hedge deja riesgo residual

    > Para cualquier $\Delta$ determinista entre rebalanceos, el término de salto de la cartera cubierta $\Pi=C-\Delta S$ es

    $$\big[C(SY)-C(S)-\Delta S(Y-1)\big]\,dN_t,$$

    > con varianza condicional estrictamente positiva: el riesgo de salto **no desaparece**.

    *Demostración.* Por el Teorema 7, $C(SY)-C(S)-\Delta S(Y-1)$ no es nulo c.s. para ningún $\Delta$. Como $N$ salta con probabilidad positiva, la varianza por unidad de tiempo es $\lambda\,\mathrm{Var}(\cdot)>0$. $\square$

    ---

    <span style="background:#6b7f6e;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">OBSERVACIÓN</span>

    # Elección de $\mathbb Q$ y griegas

    **Elección de $\mathbb Q$.** El conjunto de medidas martingala equivalentes es infinito. Criterios: mínima entropía, Esscher, utilidad exponencial, varianza mínima (definidos en el Glosario 3/4).

    - **Merton:** con riesgo de salto diversificable, los parámetros de salto bajo $\mathbb P$ y $\mathbb Q$ coinciden.
    - **Índices en la práctica:** hay prima por riesgo de salto, así que $\lambda^{\mathbb Q}\ne\lambda^{\mathbb P}$. Calibrar con retornos ($\mathbb P$) y con opciones ($\mathbb Q$) da parámetros distintos: de ahí la tarea de extensión.

    **Griegas.** $\Delta=\partial C/\partial S$ existe pero no cubre saltos; el vega respecto a $\sigma$ difiere del de BS y aparecen sensibilidades a $\lambda,\mu_J,\sigma_J$. Cálculo por diferencias finitas o *pathwise*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span style="background:#7a4fa0;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJEMPLO</span>

    # Ejemplo 8 · Cobertura delta simulada

    Vendedor de call ATM, $T=0.5$, $r=0.03$, 126 rebalanceos, 6000 trayectorias, bajo $\mathbb Q$.  
    Merton: $\sigma=0.15$, $\lambda=1$, $\mu_J=-0.15$, $\sigma_J=0.25$. Referencia: GBM $\sigma=0.20$.

    | P&L | Merton | GBM |
    |---|---|---|
    | media | −0.01 | 0.00 |
    | desv. est. | 7.60 | 0.44 |
    | cuantil 1 % | −28.4 | −1.2 |
    | asimetría | −2.50 | — |

    P&L: resultado del vendedor al vencimiento (prima + cobertura − payoff), con $\Delta$ de Merton (en el GBM, de BS). Cuantil 1 %: el 1 % de las trayectorias pierde más.

    Media ≈ 0, pero **cola izquierda enorme**: acciones no cubren el riesgo de salto (Cor. 10).
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Residuo de salto (ilustra el Teorema 7 y el Corolario 10).** El gráfico muestra
    $R(y)=\Delta S(y-1)+C(Sy)-C(S)$ para el delta del modelo y otros valores de $\Delta$: ninguno lo anula,
    porque $y\mapsto C(Sy)$ es estrictamente convexa.
    """)
    return


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
    Usa los parámetros de Merton del Ejemplo 8 (deslizadores de arriba).
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
    <span style="background:#b4372b;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">EJERCICIO</span>

    # Ejercicio 9 · Ticket de salida (individual)

    1. Escriba la fórmula de Merton e identifique el factor $e^{(r_k-r)T}$ (o, equivalente, $\lambda'$). ¿Qué pasaría si se omitiera?
    2. **Análisis.** Dos acciones con la misma IV ATM, el mismo $(\mu_J,\sigma_J)$ y distinta curvatura del smile: ¿cuál tiene mayor $\lambda$ y por qué? (use el Teo. 3 y el Cor. 3).
    3. **Transferencia.** Si el salto fuera exponencial doble (Kou) en lugar de lognormal, ¿qué cumulantes cambian y cómo afecta al smile?
    4. **Concepto.** En una línea: ¿por qué falla la cobertura perfecta bajo MJD? (Teorema 7).

    > **Metacognición (2 min).** *"El paso del problema que aún no podría explicar sin ayuda es…"*

    ---

    # Síntesis

    - Merton = **difusión + Poisson compuesto**; el compensador $\lambda m$ hace de $e^{-rt}S_t$ una martingala, y $\ln S$ es un proceso de Lévy de actividad finita.
    - Precio = **mezcla Poisson de BS**: $C_0=\sum_kP_k'\,\mathrm{BS}(r_k,\sigma_k)$, con error de truncamiento acotado por $S_0\,\mathbb P(N'>K^*)$.
    - Cumulantes $\propto T$ ⇒ asimetría y curtosis que **decaen como $T^{-1/2}$ y $T^{-1}$**.
    - **Calibrar es difícil:** $\lambda$ apenas identificado con datos diarios; la MLE converge al mejor ajuste en KL.
    - **Mercado incompleto:** el delta-hedge deja una cola izquierda de riesgo de salto.
    - Siguiente paso: Kou, Bates y calibración por Fourier.

    ---

    # Índice de resultados

    | Resultado | Enunciado en una línea |
    |---|---|
    | Def. 1–3 | Supuestos H1–H4, compensador $m$, EDS de Merton |
    | Lema 1 | $M_t=\sum(Y_i-1)-\lambda mt$ es martingala |
    | Teo. 1 | $S_t=S_0\exp[(r-\lambda m-\sigma^2/2)t+\sigma W_t+\sum J_i]$ |
    | Teo. 2 / Cor. 1–2 | $e^{-rt}S_t$ es martingala; $F_0=S_0e^{rT}$ |
    | Prop. 1 | $\ln S$ es Lévy de actividad finita |
    | Prop. 2 / Teo. 3 / Cor. 3–4 | Mezcla Poisson de normales; cumulantes $\propto T$; $\gamma_1=O(T^{-1/2})$, $\gamma_2=O(T^{-1})$ |
    | Teo. 4 | Función característica cerrada |
    | Lema 2 / Teo. 5 / Cor. 5–7 | Fórmula de Merton; $C_0=\sum P_k'\mathrm{BS}_k$; límites y paridad |
    | Prop. 3 / Cor. 8 | Error de truncamiento $\le S_0\,\mathbb P(N'>K^*)$ |
    | Def. 4 / Prop. 4 | IV; aplanamiento del smile, $\Sigma_\infty=\sqrt{\kappa_2/T}$ |
    | Def. 5 / Prop. 5 / Teo. 6 / Cor. 9 | MLE, acotación, convergencia a KL, errores estándar |
    | Def. 6 / Teo. 7 / Cor. 10 | Mercado incompleto; el delta-hedge no cubre saltos |

    ---

    # Tarea de extensión y evaluación

    **Vía A (en clase).** Calibración por MLE sobre retornos (Ejercicio 8). **Vía B (extensión).** Calibrar por mínimos cuadrados sobre precios de opciones (SPX o una acción del índice) usando $\chi_T$ (Teo. 4) y Carr–Madan, y comparar con la MLE de la Vía A (Teo. 6). **Entregable:** notebook con código, tabla de parámetros y gráfico de smile ajustado vs. empírico.

    | Componente | Puntos | Criterios clave |
    |---|---|---|
    | Ticket de salida (Ej. 9) | 4 | Factor $e^{(r_k-r)T}$, $\lambda$ vs. curvatura, Kou, incompletitud |
    | Calibración en parejas, Vía A (Ej. 8) | 6 | Código válido (2), errores estándar (1), identificación débil (2), inicios alternativos (1) |
    | Extensión, Vía B | 10 | Ajuste (3), comparación con MLE (2), gráfico (2), discusión de límites (3) |

    ---

    <span style="background:#1f6b44;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">SOLUCIÓN</span>

    # Soluciones · Ejercicios 1 a 3

    **Ej. 1.** (1) Sin compensador, $\mathbb E[S_t]=S_0e^{(r+\lambda m)t}$. (2) $m=e^{-0.08}-1=-0.0769$, $\lambda m=-0.3844$: $\mathbb E[S_1]=100\,e^{0.05-0.3844}=71.58$, frente a $105.13=S_0e^{rt}$. (3) **No** es martingala: el precio de un forward sería 32 % demasiado bajo (sesgo $e^{\lambda mt}$).

    **Ej. 2.** $\gamma_1(T)=\gamma_1(1)T^{-1/2}$, $\gamma_2(T)=\gamma_2(1)T^{-1}$. Para $T=0.25$: $\gamma_1=-0.8324$, $\gamma_2=1.736$. Para $T=4$: $\gamma_1=-0.2081$, $\gamma_2=0.1085$. La aproximación normal es mejor a $T=4$. Si $\mu_J=0$: $\kappa_3=0$ y $\kappa_4=3\lambda T\sigma_J^4>0$.

    **Ej. 3.** $\mathbb E[S_T\mid k]=S_0e^{\mu_k+s_k^2/2}$ y $\mu_k+\tfrac{s_k^2}{2}=(r-\lambda m)T-\tfrac{\sigma^2T}{2}+k\mu_J+\tfrac{\sigma^2T}{2}+\tfrac{k\sigma_J^2}{2}=(r-\lambda m)T+k\big(\mu_J+\tfrac{\sigma_J^2}{2}\big)=r_kT$. $\square$

    ---

    <span style="background:#1f6b44;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">SOLUCIÓN</span>

    # Soluciones · Ejercicios 4 a 9

    **Ej. 4.** (1) Con $\lambda'T=0.8564$: $S_0\,\mathbb P(N'>9)>10^{-6}$ y $S_0\,\mathbb P(N'>10)=2.1\cdot10^{-7}$ ⇒ $K^*=10$ para $10^{-6}$; $K^*=13$ para $10^{-10}$ (cota $5.9\cdot10^{-11}$). (2) $\mathrm{Put}_0=16.8617-100+95.1229=11.9846$.

    **Ej. 5.** Implementación libre. El error relativo de MC es mayor en $K=120$ (call OTM): el precio es pequeño y el payoff es raro, así que la varianza relativa crece; en $K=80$ (ITM) es el menor.

    **Ej. 6.** (1) $T=0.1$: $35.8-20.0=15.8$ puntos porcentuales (pp); $T=2$: $21.6-19.5=2.1$ pp: se reduce ≈ 7.5 veces. (2) $\mu_J<0$ (saltos negativos dominan). (3) El skew empírico a plazos largos decae más despacio que $T^{-1/2}$.

    **Ej. 7.** (1) $6048\,(1-e^{-5/252})\approx118.8$ días con salto. (2) $\lambda$ solo se informa por ≈ 119 eventos entre 6048 datos y se confunde con ruido; $\sigma$ se informa por los 5900+ días restantes.

    **Ej. 9.** (1) Sin $e^{(r_k-r)T}$ se rompe la martingala. (2) A igual IV ATM ($\kappa_2$ fijo) y mismo $(\mu_J,\sigma_J)$, $\gamma_2=\lambda T\,\mathbb E[J^4]/\kappa_2^2$ crece con $\lambda$: mayor curvatura ⇒ mayor $\lambda$ (Teo. 3, Cor. 3). (3) Cambian todos los $\kappa_n$ con $n\ge2$, pues dependen de $\mathbb E[J^n]$. (4) $C(SY)-C(S)$ no es lineal en $Y$ (Teo. 7).
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Verificación numérica de las soluciones** (Ejercicios 1, 2, 4 y 7), con las funciones del cuaderno.
    """)
    return


@app.cell
def _(comp_m, math, merton_call, mo, np, poisson):
    # Ejercicio 1: sin compensador
    _m1 = comp_m(-0.1, 0.2)
    _E_sin = 100 * math.exp((0.05 + 5 * _m1) * 1.0)
    _E_con = 100 * math.exp(0.05)
    # Ejercicio 2: escalamiento con T (Ejemplo 2)
    _k2 = 0.04 + 5 * (0.04 + 0.01)
    _k3 = 5 * (-0.001 - 0.012)
    _k4 = 5 * (0.0001 + 0.0024 + 0.0048)
    _g1, _g2 = _k3 / _k2**1.5, _k4 / _k2**2
    # Ejercicio 4: menor K* con la cota de Poisson (Ejemplo 3)
    _lamp = 1.0 * (1 + comp_m(-0.2, 0.3))
    _ks = np.arange(0, 40)
    _bound = 100 * poisson.sf(_ks, _lamp * 1.0)
    _k_1e6 = int(_ks[_bound < 1e-6][0])
    _k_1e10 = int(_ks[_bound < 1e-10][0])
    _C0 = merton_call(100.0, 100.0, 1.0, 0.05, 0.2, 1.0, -0.2, 0.3)
    _put = _C0 - 100 + 100 * math.exp(-0.05)
    # Ejercicio 7: días con salto esperados
    _days = 6048 * (1 - math.exp(-5 / 252))
    mo.md(
        rf"""
    | Ejercicio | Resultado calculado | Valor de la solución |
    |---|---|---|
    | 1 · $\mathbb E[S_1]$ sin compensador | {_E_sin:.2f} (frente a {_E_con:.2f}; cociente {_E_sin / _E_con:.3f}) | 71.58 vs. 105.13; 32 % más bajo |
    | 2 · $\gamma_1(T{{=}}0.25)$, $\gamma_2(T{{=}}0.25)$ | {_g1 / 0.25**0.5:.4f}, {_g2 / 0.25:.3f} | −0.8324, 1.736 |
    | 2 · $\gamma_1(T{{=}}4)$, $\gamma_2(T{{=}}4)$ | {_g1 / 4**0.5:.4f}, {_g2 / 4:.4f} | −0.2081, 0.1085 |
    | 4 · menor $K^*$ para $10^{{-6}}$ y $10^{{-10}}$ | {_k_1e6}, {_k_1e10} | 10 y 13 |
    | 4 · $\mathrm{{Put}}_0$ por paridad | {_put:.4f} (con $C_0={_C0:.4f}$) | 11.9846 |
    | 7 · días con salto esperados | {_days:.1f} | ≈ 118.8 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Anexo · Glosario y referencias

    ---

    <span style="background:#444b52;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">GLOSARIO</span>

    # Glosario 1/4 · Siglas y abreviaturas

    | Sigla | Significado |
    |---|---|
    | EDS · MJD | ecuación diferencial estocástica · *Merton jump-diffusion* (el modelo de la sesión) |
    | BS · GBM | Black–Scholes · movimiento browniano geométrico, $dS/S=r\,dt+\sigma\,dW$ (el modelo de BS, sin saltos) |
    | MC | Monte Carlo: estimar una esperanza con el promedio de muestras simuladas |
    | MLE · KL | estimador de máxima verosimilitud · divergencia de Kullback–Leibler |
    | IV | volatilidad implícita |
    | c.s. · i.i.d. | casi seguramente (con probabilidad 1) · independientes e idénticamente distribuidas |
    | fgm · TCL | función generadora de momentos $\mathbb E[e^{uX}]$ · teorema central del límite |
    | P&L · e.e. | ganancia o pérdida (*profit and loss*) · error estándar |
    | SPX · SX5E | índices S&P 500 y EURO STOXX 50 |
    | VG · CGMY · NIG | *Variance Gamma* · Carr–Geman–Madan–Yor · *Normal Inverse Gaussian*: procesos de Lévy de actividad infinita |
    | L-BFGS-B | optimizador cuasi-Newton de memoria limitada, con cotas sobre los parámetros |
    | JFE | *Journal of Financial Economics* |

    ---

    <span style="background:#444b52;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">GLOSARIO</span>

    # Glosario 2/4 · Probabilidad y procesos

    - **Poisson homogéneo** de intensidad $\lambda$: $N_0=0$, incrementos independientes y $N_t-N_s\sim\text{Poisson}(\lambda(t-s))$. **Poisson compuesto**: $\sum_{i\le N_t}J_i$, con $J_i$ i.i.d. e independientes de $N$.
    - **Martingala**: proceso $M$ con $\mathbb E[M_t\mid\mathcal F_s]=M_s$ para $s\le t$.
    - **Proceso de Lévy**: $X_0=0$, trayectorias càdlàg (continuas por la derecha, con límite por la izquierda) e incrementos independientes y estacionarios. Su **tripleta** $(b,\sigma^2,\nu)$ es deriva, varianza difusiva y medida de Lévy; si $\nu(\mathbb R)<\infty$, $\ln\mathbb E[e^{iuX_1}]=iub-\tfrac{\sigma^2u^2}{2}+\int(e^{iux}-1)\,\nu(dx)$.
    - **Medida de Lévy** $\nu(A)$: número esperado de saltos con tamaño en $A$ por unidad de tiempo. **Actividad finita / infinita**: $\nu(\mathbb R)<\infty$ / $\nu(\mathbb R)=\infty$.
    - **Exponencial de Doléans-Dade** $\mathcal E(Z)$: la solución única de $dU_t=U_{t^-}\,dZ_t$, $U_0=1$.
    - **Identidad de Wald**: si $N$ es independiente de las $Z_i$ i.i.d. e integrables, $\mathbb E\big[\sum_{i\le N}Z_i\big]=\mathbb E[N]\,\mathbb E[Z]$.
    - **Cumulantes** $\kappa_n$: coeficientes de $u^n/n!$ en $\ln\mathbb E[e^{uX}]$; $\kappa_1$ = media, $\kappa_2$ = varianza. **Asimetría** $\gamma_1=\kappa_3/\kappa_2^{3/2}$ y **exceso de curtosis** $\gamma_2=\kappa_4/\kappa_2^2$ (ambos 0 para la normal).
    - **Cota de Chernoff**: $\mathbb P(Z\ge a)\le e^{-\xi a}\,\mathbb E[e^{\xi Z}]$ para todo $\xi>0$.

    ---

    <span style="background:#444b52;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">GLOSARIO</span>

    # Glosario 3/4 · Mercado y valoración

    - **Medida martingala equivalente** (o **neutral al riesgo**): $\mathbb Q$ equivalente a $\mathbb P$ bajo la cual los precios descontados, como $e^{-rt}S_t$, son martingalas; garantiza ausencia de arbitraje.
    - **Mercado completo / incompleto**: todo payoff acotado es replicable / hay payoffs no replicables, y entonces existen infinitas $\mathbb Q$ (Def. 6).
    - **Delta-hedge**: vender una opción y cubrirla con $\Delta=\partial C/\partial S$ acciones, **rebalanceando** esa cantidad en fechas discretas.
    - **Criterios para elegir $\mathbb Q$.** *Esscher*: $d\mathbb Q/d\mathbb P\propto e^{\vartheta X_T}$, con $\vartheta$ tal que $e^{-rt}S_t$ sea martingala. *Mínima entropía*: minimiza $\mathrm{KL}(\mathbb Q\,\|\,\mathbb P)$ entre las medidas martingala. *Varianza mínima*: minimiza la varianza de $d\mathbb Q/d\mathbb P$. *Utilidad exponencial*: precio de indiferencia de un agente con $U(x)=-e^{-ax}$.
    - **Carr–Madan**: la call se obtiene como transformada de Fourier inversa de $\chi_T$, tras multiplicar su precio por $e^{\delta\ln K}$ para hacerlo integrable.
    - **Heston · Bates · Kou**: volatilidad estocástica con reversión a la media · Heston más saltos lognormales · saltos con log-salto exponencial doble (asimétrico).
    - **Smile / skew**: gráfica de la IV frente al strike $K$; el skew es su pendiente (Def. 4).

    ---

    <span style="background:#444b52;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">GLOSARIO</span>

    # Glosario 4/4 · Estadística, cómputo y clase

    - **Identificación débil**: $\ell$ es casi plana en alguna dirección del espacio paramétrico, de modo que $-\nabla^2\ell(\hat\theta)$ es casi singular. Su **número de condición** es el cociente entre su mayor y su menor autovalor.
    - **Verosimilitud perfilada** de $\lambda$: $\ell_p(\lambda)=\max_{\sigma,\mu_J,\sigma_J}\ell(\theta)$ con $\lambda$ fijo (Cor. 9).
    - **Inicios múltiples**: repetir el optimizador desde varios $\theta$ iniciales para detectar óptimos locales.
    - **`logsumexp`**: $\ln\sum_k e^{a_k}=a^*+\ln\sum_k e^{a_k-a^*}$ con $a^*=\max_k a_k$; evita desbordamientos numéricos.
    - **Semilla fija**: valor inicial del generador aleatorio; hace reproducibles las simulaciones.
    - **Ticket de salida**: actividad individual y breve al cierre de la clase para comprobar lo aprendido.
    - **Hora 1 / Hora 2**: los dos bloques de 60 min de la sesión (teoría; simulación); los **bloques** de la Ruta se numeran dentro de cada hora.

    ---

    <span style="background:#444b52;color:#fff;padding:2px 10px;border-radius:5px;font-size:.7em;font-weight:700;letter-spacing:.12em">REFERENCIAS</span>

    # Referencias

    - Black, F. y Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
    - Merton, R. C. (1973). Theory of rational option pricing. *Bell Journal of Economics and Management Science*, 4(1), 141–183.
    - Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
    - Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327–343.
    - Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes implicit in Deutsche mark options. *Review of Financial Studies*, 9(1), 69–107.
    - Barndorff-Nielsen, O. E. (1997). Normal inverse Gaussian distributions and stochastic volatility modelling. *Scandinavian Journal of Statistics*, 24(1), 1–13.
    - Madan, D. B., Carr, P. y Chang, E. C. (1998). The variance gamma process and option pricing. *European Finance Review*, 2(1), 79–105.
    - Carr, P. y Madan, D. (1999). Option valuation using the fast Fourier transform. *Journal of Computational Finance*, 2(4), 61–73.
    - Duffie, D., Pan, J. y Singleton, K. (2000). Transform analysis and asset pricing for affine jump-diffusions. *Econometrica*, 68(6), 1343–1376.
    - Kou, S. G. (2002). A jump-diffusion model for option pricing. *Management Science*, 48(8), 1086–1101.
    - Carr, P., Geman, H., Madan, D. B. y Yor, M. (2002). The fine structure of asset returns: an empirical investigation. *Journal of Business*, 75(2), 305–332.
    - White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
    """)
    return



if __name__ == "__main__":
    app.run()
