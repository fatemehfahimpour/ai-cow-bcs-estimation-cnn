import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns


# site link: https://zenodo.org/records/20290988

df = pd.read_csv('Dataset.csv')
output_dir = ''

# def get_path_prefix(path):
#     if pd.isna(path):
#         return "unknown"
#
#     path = str(path).strip()
#
#     first_part = re.split(r"[\\/]", path)[0]
#
#     return first_part
#
#
# df["path_prefix"] = df["RGB_path"].apply(get_path_prefix)
#
# for prefix, group in df.groupby("path_prefix"):
#     group = group.drop(columns=["path_prefix"])
#
#     output_file = f"{output_dir}/{prefix}.csv"
#
#     group.to_csv(output_file, index=False, encoding="utf-8-sig")
#
#     print(f"{len(group)} ردیف ذخیره شد در: {output_file}")


sns.set_theme(style="whitegrid")

dist_output_dir = 'distributions'

if not os.path.exists(dist_output_dir):
    os.makedirs(dist_output_dir)

csv_files = glob.glob(os.path.join(output_dir, "*.csv"))

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    label = file_name.replace('.csv', '')

    try:
        df = pd.read_csv(file_path)

        if 'BCS' in df.columns:
            bcs_values = pd.to_numeric(df['BCS'], errors='coerce').dropna()

            if not bcs_values.empty:
                plt.figure(figsize=(8, 5))

                sns.histplot(bcs_values, kde=True, color='skyblue', edgecolor='black')

                plt.title(f'BCS Distribution - {label}', fontsize=14)
                plt.xlabel('BCS Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)

                save_path = os.path.join(dist_output_dir, f"{label}_dist.png")
                plt.savefig(save_path, dpi=200)
                plt.close()

    except Exception as e:
        pass

