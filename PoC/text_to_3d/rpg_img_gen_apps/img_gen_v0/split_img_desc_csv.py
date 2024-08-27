# All Rights Reserved
# 
# Copyright © 2024 Patrick Devaney
# 
# THE CONTENTS OF THIS PROJECT ARE PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED COPYING, TRANSFERRING, OR REPRODUCTION OF THE CONTENTS OF THIS PROJECT, VIA ANY MEDIUM, IS STRICTLY PROHIBITED.
# 
# The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them for any purpose other than the purpose for which they were provided to you.
# 
# The software is provided “AS IS”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.

import pandas as pd

# Function to split the input CSV into smaller CSV files with 200 rows each
def split_csv(input_csv, rows_per_split=200):
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    num_splits = (total_rows // rows_per_split) + 1
    for i in range(num_splits):
        split_df = df[i*rows_per_split:(i+1)*rows_per_split]
        split_df.to_csv(f'img_desc_input_csv_split_{i+1:02d}.csv', index=False)
    return num_splits

# Split the input CSV
input_csv = 'descriptions.csv'
split_csv(input_csv)