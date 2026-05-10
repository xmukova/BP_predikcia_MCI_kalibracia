# ZAKLAD PRE VYBER PACIENTOV, KTORY SPLNAJU INKLUZIVNE KRITERIA
import pandas as pd
TIME_HORIZONT = 5

# nacitanie "surovych" dat, ktore obsahuje vsetkych pacientov, ktory su v NACC databaze
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep='\t')
    print("Data shape: ", data.shape)
    return data

# filtrovanie dat podla inkluzivnych kriterii
def prepare_for_eda(data: pd.DataFrame) -> pd.DataFrame:
    print("Počet všetkých pacientov pred vyberom pacientov vhodnych pre moju studiu: ", data.shape[0])
    a_patients = data[ (data["TIME"] <= TIME_HORIZONT) & (data["EVENT_MCI"] == 1) ]     
    b_patients = data[ (data["TIME"] > TIME_HORIZONT) & (data["EVENT_MCI"] == 1) ]      # zmenit event_mci -> 0
    c_patients = data[ (data["TIME"] >= TIME_HORIZONT) & (data["EVENT_MCI"] == 0) ]
    d_patients = data[ (data["TIME"] < TIME_HORIZONT) & (data["EVENT_MCI"] == 0) ]      # vyhodit -> sledovani kratko
    
    # Vyhodenie pacientov kt. boli sledovani kratsie ako 4 roky
    data_bez_d = data.drop(d_patients.index)
    data = data_bez_d.copy()

    # Definicia cielovej premennej so spravnou udalostou
    data["target"] = 0
    data.loc[data["NACCID"].isin(a_patients["NACCID"]), "target"] = 1

    print(f"Počet pacientov, ktorí prešli do MCI DO {TIME_HORIZONT} rokov: ", a_patients.shape[0])
    print(f"Počet pacientov, ktorí prešli do MCI PO {TIME_HORIZONT} rokoch: ", b_patients.shape[0], " (pre nás to znamená, že do {TIME_HORIZONT} rokov NEPREŠLI)")
    print(f"Počet pacientov, ktorí neprešli do MCI a sledujú sa minimálne {TIME_HORIZONT} rokov: ", c_patients.shape[0])
    print(f"Počet pacientov, ktorí neprešli do MCI a sledujú sa menej ako {TIME_HORIZONT} rokov: ", d_patients.shape[0], f"(pre nás nerelevantný - nevieme, ako dopadnú)")

    return data

DATA_RAW_PATH = "data/raw/nacc_data_2025.csv"
DATA_EDA_PATH = "data/ready_for_EDA/nacc_ready_for_eda_2025_HORIZONT_5.csv"

def main():
    df = load_data(DATA_RAW_PATH)                   # nacitanie datasetu
    print("----------- Raw dataset je nacitany")

    df_eda = prepare_for_eda(df)                    # priprava datasetu a filtrovanie pacientov, ktory su relevantny pre moju pracu
    print("----------- Dataset je pripraveny na Exploratory Data Analysis")

    df_eda.to_csv(DATA_EDA_PATH, index=False)       # ulozenie datasetu, ktory je pripraveny na EDA
    print(f"----------- Dataset ulozeny do {DATA_EDA_PATH}")
    
if __name__ == "__main__":
    main() 