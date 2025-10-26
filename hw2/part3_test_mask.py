import pandas as pd
EXCEL_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/color_coding_semantic_segmentation_classes.xlsx"

def load_semantic_ID_table(excel_path):
    df = pd.read_excel(excel_path)
    print(df.columns)
    id_col = None
    name_col = None

    # 嘗試自動判斷欄位
    for c in df.columns:
        if c.strip().lower() in ["id", "class_id", "semantic_id", "label_id", "index"]:
            id_col = c
        elif c.strip().lower() in ["name", "class", "object", "label", "color name"]:
            name_col = c

    if id_col is None or name_col is None:
        raise ValueError(f"❌ 無法在 Excel 中找到 ID 或名稱欄位，檢查欄名：{list(df.columns)}")

    id_map = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip().lower()
        try:
            id_val = int(row[id_col])
        except ValueError:
            continue
        id_map[name] = id_val

    print(f"[INFO] 成功載入 {len(id_map)} 個語意分類。")
    return id_map

id_map = load_semantic_ID_table(EXCEL_PATH)
print(id_map)