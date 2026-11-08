<h1>decision-support-and-forecasting-toolkit</h1>
<p>Quantitative decision-making, time-series forecasting algorithms, econometric OLS regression, and operations research modeling in Python</p>

<h2>Project description</h2>
<p>This repository provides a comprehensive computational intelligence and operations research suite covering fundamental business analytics and quantitative forecasting paradigms:</p>
<ol>
  <li>
    <dl>
      <dt><b>Time Series Forecasting &amp; Error Diagnostics:</b></dt>
      <dd>Implements Simple Moving Average (SMA) and Weighted Moving Average (WMA) with customizable weight vectors.</dd>
      <dd>Applies Single Exponential Smoothing (SES) and double-parameter Holt's Linear Trend smoothing ($\alpha, \beta$) for multi-step horizon forecasting.</dd>
      <dd>Provides polynomial interpolation via Lagrange basis polynomials to estimate missing historical demand observations.</dd>
      <dd>Measures forecasting accuracy using Mean Absolute Deviation (MAD) and Mean Squared Error (MSE).</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Econometric OLS Regression:</b></dt>
      <dd>Calculates ordinary least squares regression parameters ($\beta_0, \beta_1$) from first principles without external statistical libraries.</dd>
      <dd>Forecasts asset depreciation schedules based on usage metrics.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Multi-Criteria Decision Analysis (MCDA):</b></dt>
      <dd>Implements the Weighted Sum Model (WSM) via vector-matrix dot products ($S = R \cdot w$) for alternative ranking across heterogeneous criteria.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Transportation &amp; Logistics Optimization:</b></dt>
      <dd>Models classical balanced transportation problem matrices under strict supply and demand conservation constraints.</dd>
    </dl>
  </li>
</ol>

<h2>Project files</h2>
<table>
  <tr>
    <th colspan="2">decision-support-and-forecasting-toolkit</th>
  </tr>
  <tr>
    <td>main.py</td>
    <td>Master CLI orchestrator running the full suite or individual algorithmic modules</td>
  </tr>
  <tr>
    <td>modules/forecasting.py</td>
    <td>Time-series prediction routines (SMA, WMA, SES, Holt Trend, Lagrange, MAD/MSE)</td>
  </tr>
  <tr>
    <td>modules/regression.py</td>
    <td>Ordinary Least Squares analytical regression engine</td>
  </tr>
  <tr>
    <td>modules/mcda.py</td>
    <td>Multi-criteria scoring model and matrix-based alternative evaluation</td>
  </tr>
  <tr>
    <td>modules/transportation.py</td>
    <td>Transportation problem matrix generator and cost evaluation engine</td>
  </tr>
  <tr>
    <td>README.md</td>
    <td>Project documentation and mathematical formulations</td>
  </tr>
  <tr>
    <td>.gitignore</td>
    <td>Exclusion rules for bytecode, virtual environments, and local runtime files</td>
  </tr>
</table>

<h2>Methodological Overview</h2>
<table>
  <tr>
    <th>Category</th>
    <th>Method / Algorithm</th>
    <th>Mathematical Formulation</th>
  </tr>
  <tr>
    <td><b>Forecasting</b></td>
    <td>Holt's Linear Trend</td>
    <td>$S_t = \alpha y_t + (1-\alpha)(S_{t-1} + T_{t-1})$, $T_t = \beta(S_t - S_{t-1}) + (1-\beta)T_{t-1}$</td>
  </tr>
  <tr>
    <td><b>Forecasting</b></td>
    <td>Lagrange Interpolation</td>
    <td>$P(x) = \sum_{i=0}^n y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$</td>
  </tr>
  <tr>
    <td><b>Evaluation</b></td>
    <td>MAD &amp; MSE</td>
    <td>$\text{MAD} = \frac{1}{n} \sum |y_t - \hat{y}_t|, \quad \text{MSE} = \frac{1}{n} \sum (y_t - \hat{y}_t)^2$</td>
  </tr>
  <tr>
    <td><b>Regression</b></td>
    <td>OLS Slope &amp; Intercept</td>
    <td>$\beta_1 = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}, \quad \beta_0 = \bar{y} - \beta_1 \bar{x}$</td>
  </tr>
  <tr>
    <td><b>Decision Making</b></td>
    <td>Weighted Sum Model</td>
    <td>$S_i = \sum_{j=1}^m r_{ij} \cdot w_j \quad (S = R \cdot w)$</td>
  </tr>
  <tr>
    <td><b>Logistics</b></td>
    <td>Transportation Problem</td>
    <td>$\min \sum_{i} \sum_{j} c_{ij} x_{ij} \quad \text{s.t.} \sum_j x_{ij} = s_i, \sum_i x_{ij} = d_j$</td>
  </tr>
</table>

<h2>Installation and launch</h2>
<ol>
  <b>
    <li>
      <dl>
        <dt>Clone repository</dt>
        <dd>
          <code>git clone https://github.com/Kyoryuu-lab/decision-support-and-forecasting-toolkit.git</code>
        </dd>
        <dd>
          <code>cd decision-support-and-forecasting-toolkit</code>
        </dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Install dependencies</dt>
        <dd>
          <code>pip install numpy matplotlib</code>
        </dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Run complete analytics suite</dt>
        <dd>
          <code>python main.py --module all</code>
        </dd>
        <dd>Run specific module: <code>python main.py --module forecasting</code> (Options: <code>forecasting</code>, <code>regression</code>, <code>mcda</code>, <code>transport</code>)</dd>
      </dl>
    </li>
  </b>
</ol>

<h2>Example of console output (python main.py --module all)</h2>
<samp>
  <p>
    ===========================================================================<br />
    &nbsp;&nbsp;1. MODUL: ZEITREIHENPROGNOSE &amp; GUETEMASSE<br />
    ===========================================================================<br />
    [*] Lagrange-Interpolation (Tag x=2): 150.00<br />
    [*] Gleitender Durchschnitt (SMA, n=3) : 110.00<br />
    [*] Gewichteter Durchschnitt (WMA, k=3): 106.00<br />
    [*] Exponentielle Glaettung (SES, alpha=0.3): 130.49<br />
    [*] Holt-Prognose (Monate 13, 14, 15): 250.77, 261.54, 272.31<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- Modell SMA&nbsp;&nbsp;| MAD:&nbsp;&nbsp;6.00 | MSE:&nbsp;&nbsp;47.50<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- Modell WMA&nbsp;&nbsp;| MAD:&nbsp;&nbsp;5.25 | MSE:&nbsp;&nbsp;36.50<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- Modell SES&nbsp;&nbsp;| MAD:&nbsp;&nbsp;4.72 | MSE:&nbsp;&nbsp;28.93<br />
    <br />
    ===========================================================================<br />
    &nbsp;&nbsp;2. MODUL: OEKONOMETRISCHE OLS-REGRESSION<br />
    ===========================================================================<br />
    [*] Regressionsgerade: Preis = 35.588 + (-0.1176) * Laufleistung<br />
    [*] Prognose fuer 150.000 km: 17.94 Tsd. USD<br />
    <br />
    ===========================================================================<br />
    &nbsp;&nbsp;3. MODUL: MULTIKRITERIELLE ENTSCHEIDUNGSANALYSE (MCDA)<br />
    ===========================================================================<br />
    [*] Rangfolge der Alternativen (Nutzwertanalyse):<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Platz 1: MacBook Pro&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Gesamtscore: 7.600<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Platz 2: Dell XPS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Gesamtscore: 7.600<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Platz 3: Lenovo ThinkPad&nbsp;&nbsp;| Gesamtscore: 7.800<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Platz 4: HP Spectre&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Gesamtscore: 7.200<br />
    <br />
    ===========================================================================<br />
    &nbsp;&nbsp;4. MODUL: TRANSPORTOPTIMIERUNG<br />
    ===========================================================================<br />
    [*] Beste ermittelte Zuteilungsmatrix:<br />
    [[100&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;0]<br />
    &nbsp;[200 100&nbsp;&nbsp;&nbsp;0]<br />
    &nbsp;[&nbsp;&nbsp;0 100 200]]<br />
    [*] Minimale Gesamtkosten: 3400.00<br />
    <br />
    ===========================================================================<br />
    &nbsp;&nbsp;AUSFUEHRUNG ERFOLGREICH ABGESCHLOSSEN.<br />
    ===========================================================================<br />
  </p>
