import pandas as pd

def preprocess_input(df):
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Weekday'] = df['Order Date'].dt.weekday
        df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)

    if 'PromotionFlag' not in df.columns:
        df['PromotionFlag'] = 0

    try:
        with open('training_columns.txt', 'r') as f:
            training_columns = f.read().split(',')
    except Exception as e:
        raise ValueError(f"Failed to load training columns: {e}")

    final_data = df.reindex(columns=training_columns, fill_value=0)

    return final_data
