
import re

def replace_parquet_with_csv():
    with open('main.tex', 'r') as f:
        content = f.read()

    # Regex to find pd.read_parquet('filename.parquet') or pd.read_parquet("filename.parquet")
    # and replace it with pd.read_csv('filename.csv')
    pattern = r"pd\.read_parquet\(['\"](.+?)\.parquet['\"]\)"
    replacement = r"pd.read_csv('\1.csv')"
    
    new_content = re.sub(pattern, replacement, content)

    with open('main.tex', 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    replace_parquet_with_csv()
