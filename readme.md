# Stock Portfolio Optimizer

## Optimize your portfolio for higher profit and more stable trends.

This app optimizes the weights of stocks in a portfolio. It maximizes both of capital gain and stability of portfolio price.
The balance between these two factors are decided by alpha parameter.

# Technical detail about the optimization

```math
\underset{w} {\text{minimize}} L(w) = \underset{w} {\text{minimize}} -\left( \text{capital gain}(w) \right) + \alpha \left( \text{fluctuation}(w)  \right)\\
\text{capital gain}(w) = s(m-1)\
\text{fluctuation}(w) = \sqrt{\frac{\sum_{i}(d_i - \frac{\sum d_i}{m-1})^2}{m-2}}<br>
d = \{ r_i - r_{i-1} \mid 1 \leq i \leq m-1 \}  
r_j = q_j - l_j
l_j = sj + t
s = \frac{\sum_{j} (q_j - c_y)(j - c_x)}{\sum_{j} (j - c_x)^2}, t = \frac{c_y - s}{c_x}
(c_x = \frac{\sum_{j} j}{m}, c_y = \frac{\sum_{j} q_j}{m})
q_j = \sum_{i=0}^{n-1} w_i p_j^i
```

where $n$ is the number of stocks in the portfolio, $m$ is the number of records, which is the number of days, in the training dataset, $w$ is the weights of stocks, $p$ is the prices of stocks, $q$ is the $l$ is the regression line of the portfolio price, $s$ is its slope, $t$ is its intercept, $r$ is the fluctuation of the stock prices on the regression lines, $d$ is the daily fluctuation of the stock prices, $\text{fluctuation}$ is the unbiased standard deviation of the daily fluctuation and $\text{capital gain}$ is the increase of the stock price on the regression line.

# Webapp

📈Stock Portfolio Optimizer  
https://dixhom-stock-portfolio-optimizer-main-51ft6w.streamlit.app/
