import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt

# 1. Datasets
orion_df = pd.read_csv('data/orion_eod.csv')
cleanergy_df = pd.read_csv('data/cleanergy_eod.csv')
novaterra_df = pd.read_csv('data/novaterra_eod.csv')

def plot_structural_breaks(df, break_indices, company_name):
    colors = {
    "price": "#154360",      # dark navy
    "break_line": "#2471A3", # medium blue
    "break_point": "#5499C7" # light blue
}

    fig, ax = plt.subplots(figsize=(14, 7))

    # Stock price
    ax.plot(
        df['Date'],
        df['Close'],
        linewidth=2,
        color= colors['price'], 
        label='Stock Price',
    )

    # Structural breaks
    for idx in break_indices:

        # Ignore the last point returned by ruptures
        if idx < len(df):

            break_date = df.iloc[idx]['Date']
            break_price = df.iloc[idx]['Close']

            ax.axvline(
                x=break_date,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
                color=colors['break_line']
            )

            ax.scatter(
                break_date,
                break_price,
                s=60,
                zorder=5,
                color= colors['break_point']
            )

            ax.annotate(
                break_date.strftime('%Y-%m-%d'),
                xy=(break_date, break_price),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                rotation=45
            )

    ax.set_title(
        f'{company_name} - Structural Break Detection',
        fontsize=16,
        fontweight='bold'
    )

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def find_structural_breaks(df, company_name, penalty=10):

    df['Date'] = pd.to_datetime(df['Date'])

    # Filter 2022 onwards
    df = df[df['Date'] >= '2022-01-01'].reset_index(drop=True)

    # Convert prices to a 1D numpy array for the algorithm
    signal = df['Close'].values

    # 3. Initialize and fit the PELT algorithm
    algo = rpt.Pelt(model="rbf").fit(signal)

    # Predict the break points using the penalty
    break_indices = algo.predict(pen=penalty)

    # 4. Map the indices back to actual dates
    print(f"--- Structural Breaks for {company_name} ---")
    break_dates = []

    for i in break_indices:
        # ruptures returns the end index as the final break, we ignore it
        if i < len(df):
            break_date = df.iloc[i]['Date'].strftime('%Y-%m-%d')
            break_dates.append(break_date)
            print(f"Break detected on: {break_date}")

    print("\n")
    return df, signal, break_indices


# Run the algorithm for all three companies
orion_data, orion_sig, orion_breaks = find_structural_breaks(orion_df, "Orion Chips", penalty=20)
cleanergy_data, clean_sig, clean_breaks = find_structural_breaks(cleanergy_df, "Cleanergy", penalty=20)
nova_data, nova_sig, nova_breaks = find_structural_breaks(novaterra_df, "NovaTerra AI", penalty=20)

plot_structural_breaks( orion_data, orion_breaks,"Orion Chips")
plot_structural_breaks(cleanergy_data,clean_breaks,"Cleanergy")
plot_structural_breaks( nova_data,nova_breaks,"NovaTerra AI")