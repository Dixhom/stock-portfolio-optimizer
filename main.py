# -*- coding: utf-8 -*-

# v def stocknames cached
# v def optimize external class returns props
# v alpha textbox
# v df.plot trainset testset one graph vertical line
# v props * assets show
# v train 9 month + test 3 month
# v at least 1 year if not show stocks shorter than 1 year
# show final profit

import streamlit as st
import pandas as pd
import numpy as np
import optimize as opt
import matplotlib.pyplot as plt


@st.cache
def load_nasdaq_data():
    return pd.read_pickle('nasdaq.pickle')


def draw_graphs(plot_area, df, test_vline):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # draw each series
    colors = ['#a9a9a9'] * (df.shape[1] - 1) + ['#ff0000']
    for column, color in zip(df.columns, colors):
        axes[0].plot(df[column], label=column)
        axes[1].plot(df[column], color=color, label=column)

    # legend
    axes[0].legend(loc='upper left')

    # axis label rotation
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[1].tick_params(axis='x', labelrotation=45)

    # axis labels
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Stock price [USD]")

    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Stock price [USD]")

    # a line to indicate the start of the test dataset
    ymin = df.min().min()
    ymax = df.max().max()
    axes[1].vlines(x=test_vline, ymin=ymin, ymax=ymax,
                   colors='blue', ls=':', lw=2, label='vline_single')
    # draw the graph in the placeholder
    plot_area.pyplot(fig)


def main():
    st.title('📈Stock Portfolio Optimizer')
    st.text('Optimize your portfolio for higher profit and more stable trends.')
    assets = st.number_input(label='💵How much money will you invest in dollars?',
                             value=10000,
                             )

    # load nasdaq stock data
    nasdaq = load_nasdaq_data()
    # list of stocks
    options = nasdaq.symbol.to_list()
    # multi selection
    selected_stocks = st.multiselect(
        '🏦Which stocks do you want to buy? (NASDAQ)', options, default=['MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX'])

    # get a dataframe for selected stocks
    if len(selected_stocks) > 0:
        df_selected_stocks = nasdaq.copy()
        df_selected_stocks = df_selected_stocks[df_selected_stocks.symbol.isin(
            selected_stocks)][['symbol', 'name', 'country', 'ipoyear', 'sector', 'industry']]
        st.dataframe(df_selected_stocks)

    # alpha
    alpha = st.number_input(label="""(Optional) Input the balance between the profit and the stability.
The value takes the value from 0 to infinity.
The higher the value is, the more stable the stock price is.""",
                            value=0.01)

    # execute button
    optimize_btn_ph = st.empty()
    optimize_btn = optimize_btn_ph.button('Optimize', disabled=False, key='1')
    text_wait = st.empty()
    if optimize_btn and selected_stocks:
        optimize_btn_ph.button('Optimize', disabled=True, key='2')
        text_wait.markdown('**Optimizing...**')
        df, prop, test_vline = opt.optimize_portfolio(selected_stocks, alpha)
        optimize_btn_ph.button('Optimize', disabled=False, key='3')
        text_wait.empty()

        # prepare plot
        st.header('The graphs of stock price trends')
        plot_area = st.empty()
        st.text("""The red series on the right graph is the total sum of the entire portfolio.
The weights of the stocks were optimized using the data on the left of the blue dotted vertical line.
The data on the right are the unseen ones by the optimized weights.""")

        # draw plots
        draw_graphs(plot_area, df, test_vline)

        # optimal amount for each stock
        st.header('How much to buy for each stock [USD]')
        amounts = list(map(int, prop * assets))
        amounts += [sum(amounts)]
        profits = df.iloc[-1, :] - df.iloc[0, :]
        stock_amount = pd.DataFrame(
            {'Stock': selected_stocks + ['total'],
             'How much to buy [USD]': amounts,
             'Profit': profits
             })
        st.dataframe(stock_amount)
        st.text("The sum of the amounts might not be equal to the total investment because each of them was rounded down.")


if __name__ == '__main__':
    main()
