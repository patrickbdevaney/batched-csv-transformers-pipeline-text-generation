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

import csv

def stringify_csv(input_file, output_file):
    try:
        with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            
            for row in reader:
                writer.writerow(row)
                
        print(f"Stringified CSV file saved to {output_file}")
    except csv.Error as e:
        print(f"Error processing CSV file: {e}")

# Example usage
input_file = 'fantasydesc.csv'
output_file = 'stringified_fantasydesc.csv'
stringify_csv(input_file, output_file)