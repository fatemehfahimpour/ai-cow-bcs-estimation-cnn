import os
import urllib.parse
import pandas as pd
import requests
from bs4 import BeautifulSoup

base_url = "https://iplab.dmi.unict.it/legacy/BCS/"
url = base_url + "dataset.html"
MAIN_PATH = '../BCS_dataset'

save_dir = "../BCS_dataset/images"
os.makedirs(save_dir, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

dataset_labels = []

tables = soup.find_all("table")
for idx , table in enumerate(tables):
    cow_img_tag = table.find("img", src = lambda s: s and "DatasetImagesCows" in s)

    if cow_img_tag:
        img_src = cow_img_tag.get("src")

        # in this site information is in first tr
        first_tr = table.find("tr")
        tds = first_tr.find_all("td") if first_tr else []

        # td[0]: cows image, td[1]: shape of cow, td[2]: download link
        # td[3]: BCS1, td[4]: BCS2
        if len(tds) >=5:
            try:
                bcs1_text = tds[3].get_text(strip=True)
                bcs2_text = tds[4].get_text(strip=True)

                bcs1 = float(bcs1_text)
                bcs2 = float(bcs2_text)
                bcs_mean = round((bcs1 + bcs2) / 2, 2)

                original_filename = os.path.basename(img_src)
                img_save_path = os.path.join(save_dir, original_filename)

                img_full_url = urllib.parse.urljoin(base_url, img_src)

                img_data = requests.get(img_full_url, headers=headers).content
                with open(img_save_path, "wb") as f:
                    f.write(img_data)

                dataset_labels.append({
                    "image_filename": original_filename,
                    "bcs_1": bcs1,
                    "bcs_2": bcs2,
                    "bcs_mean": bcs_mean,
                    "local_path": img_save_path,
                })

                print(f"image in table {idx} saved")

            except:
                continue


df = pd.DataFrame(dataset_labels)
df.to_csv(f"{MAIN_PATH}/labels.csv", index=False)




