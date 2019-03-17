from imbalance_data_structures import get_dollar_imbalance_bars
import pandas as pd
df = get_dollar_imbalance_bars('sample.csv', 10000, 5, 10)
df.to_csv('result_batch.csv', index=False)
