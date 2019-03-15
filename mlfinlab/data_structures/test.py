from imbalance_data_structures import get_dollar_imbalance_bars
import pandas as pd
df = get_dollar_imbalance_bars('sample.csv', 100, 200, 400)
df.to_csv('result.csv', index=False)
