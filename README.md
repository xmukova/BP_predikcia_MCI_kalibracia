# BP_predikcia_MCI_kalibracia
## Metódy kalibrácie umelej inteligencie v diagnostike ľudských chorôb
Bakalárska práca je zameraná na predikciu konverzie do aMCI v horizonte 5 rokov s dôrazom na kalibračnú kvalitu pravdepodobnostných výstupov. 

Porovnávané prediktívne modely: Logistická regresia, SVM a XGBoost.

Porovnávané kalibračné metódy: Platt scaling, isotonická regresia, temperature scaling, spline-based kalibrácia, beta kalibrácia

---

## Požiadavky
- Python 3.13
- pip
- Git (alebo digitálne médium)

---

## Inštalácia

**1. Klonovanie repozitára**
```bash
git clone <https://github.com/xmukova/BP_MCI_predikcia_kalibracia>
cd BP_IMPLEMENTACIA
```
Alebo stiahnutie digitálneho média `BP_AlenaMukova.zip`

**2. Vytvorenie virtuálneho prostredia**
```bash
python -m venv venv
```

**3. Aktivácia virtuálneho prostredia**

Windows:
```bash
venv\Scripts\activate
```
Linux / macOS:
```bash
source venv/bin/activate
```

**4. Inštalácia závislostí**
```bash
pip install -r requirements.txt
```

---

## Dáta
Práca používa dáta pochádzajúce z verejne dostupnej databázy **National Alzheimer's Coordinating Center (NACC)**. Dáta sú dostupné na vyžiadanie prostredníctvom oficiálnej stránky [https://naccdata.org/](https://naccdata.org/).

Konkrétne dáta použité v tejto práci pochádzajú z mája 2025. Dáta boli sprístupnené cez Ing. Martinu Billichovú, ktorá vykonala prvotnú úpravu dát a vytvorila premennú indikujúcu prechod a čas pacienta do aMCI - postup konzultovala s odborníkom na kognitívne testovanie a výskum, Dr. Davide Brunom z Liverpool John Moores University, UK. Pôvodný longtitudálny charakter dát bol odstránený a pre každého pacienta vytvorený 1 záznam s informáciami z prvej návštevy. 
 
Vzhľadom na to, že prístup k dátam NACC vyžaduje registráciu a schválenie žiadosti, dáta nie sú zdieľané verejne na GitHube. Pre reprodukciu experimentov je potrebné požiadať o prístup k dátovým súborom na portáli NACC. Pre potreby bakalárskej práce sú dáta priložené v digitálnom médiu v súbore
```
data/raw/nacc\_data\_2025.csv
```

Výber použitých premenných zahrnutých na modelovanie je definovaný v skripte 
```
src/config\_based\_on\_EDA.py
```


Po získaní dát umiestnite dátový súbor do:
```
data/raw/nacc_data_2025.csv
```

---

## Spustenie

Skripty spúšťajte v nasledujúcom poradí:

| Krok | Súbor | Popis |
|------|-------|-------|
| 1 | `src/data_loading_for_EDA.py` | Načítanie a filtrovanie dát podľa inkluzívnych kritérií |
| 2 | `jupyter_notebooks/EDA.ipynb` | Exploratívna analýza dát |
| 3 | `src/hyperparameter_tuning.py` | Ladenie hyperparametrov modelov |
| 4 | `src/train_model.py` | Tréning modelov |
| 5 | `jupyter_notebooks/DeLong_test.ipynb` | Štatistické porovnanie AUC ROC |
| 6 | `jupyter_notebooks/calibration_PRED.ipynb` | Analýza kalibračnej kvality |
| 7 | `src/calibration_porovnanie.py` | Porovnanie kalibračných techník |
| 8 | `src/final_model.py` | Uloženie finálnych modelov |

Vizualizácie:
```bash
python src/vizualizacia_features_x_method.py
python src/vizualizacia_k_features.py
python src/vizualizacia_tresholds.py
python src/interpretacia_LogReg.py
python src/rozdiel_k_features.py
python src/interpretacia_LogReg.py
```

---

## Sledovanie experimentov (MLflow)
 
Konfigurácie všetkých vykonaných experimentov sú definované v súbore `src/config_experiments.py`.
 
Experimenty je možné zobraziť dvoma spôsobmi:
 
**Možnosť A — načítať hotové experimenty z digitálneho média**
 
Skopírujte súbory `mlflow.db` a `mlruns/` z digitálneho média bakalárskej práce do koreňového adresára projektu, potom spustite:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Rozhranie je dostupné na `http://localhost:5000`.
 
**Možnosť B — spustite vlastné experimenty**
 
Spustite si vlastné experimenty definovaná v `src/config_experiments.py` pomocou skriptov  `src/train_model.py`, `src/hyperparameter_tuning.py` a `src/final_model.py`
 
| Experiment | Popis |
|------------|-------|
| NACC - finalne modely s kalibráciou | Finálne modely vrátane kalibrácie |
| BP:NACC - finalne hodnotenie + kalibracia | Vyhodnotenie diskriminačnej a kalibračnej kvality |
| BP:NACC - hyperparameter tuning | Ladenie hyperparametrov |
| BP:NACC - nastavenie k_features | Porovnanie počtu premenných a metód výberu |
| BP:NACC - experimenty manual z configu | Prvotné manuálne experimenty |
 
---

## Štruktúra projektu

```
BP_IMPLEMENTACIA/
├── requirements.txt                    # Zoznam závislostí
├── README.md                           # Tento súbor
│
├── data/
│   ├── raw/                            # Pôvodné nespracované dáta
│   └── ready_for_EDA/                  # Dáta pripravené na použitie
│
├── jupyter_notebooks/
│   ├── EDA.ipynb                       # Exploratívna analýza dát
│   ├── calibration_analyza_PRED.ipynb  # Analýza kalibrácie
│   └── DeLong_test.ipynb               # DeLongov test pre AUC ROC
│
├── models/
│   ├── lr_FINAL_MODEL.pkl              # Finálny model LR
│   ├── lr_pipeline.pkl                 # Pipeline LR
│   ├── lr_calibrator.pkl               # Kalibrátor LR
│   ├── svm_FINAL_MODEL.pkl             # Finálny model SVM
│   ├── svm_pipeline.pkl                # Pipeline SVM
│   ├── xgboost_FINAL_MODEL.pkl         # Finálny model XGBoost
│   ├── xgboost_pipeline.pkl            # Pipeline XGBoost
│   └── xgboost_calibrator.pkl          # Kalibrátor XGBoost
│
└── src/
    ├── config_based_on_EDA.py          # Definícia použitých premenných
    ├── config_experiments.py           # Konfigurácia experimentov
    ├── data_loading_for_EDA.py         # Načítanie a príprava dát
    ├── pipeline.py                     # Základná pipeline
    ├── pipeline_oversampling.py        # Pipeline s prevzorkovaním
    ├── hyperparameter_tuning.py        # Ladenie hyperparametrov
    ├── train_model.py                  # Tréning modelov
    ├── train_model_oversampling.py     # Tréning s prevzorkovaním
    ├── final_model.py                  # Finálne modely
    ├── calibratiob_porovnanie.py       # Kalibračné techniky
    ├── vizualizacia_calibration.py     # Vizualizácia kalibračných kriviek
    ├── vizualizacia_features_x_method.py  # Premenné naprieč metódami výberu
    ├── vizualizacia_k_features.py      # Vplyv počtu premenných na výkonnosť
    ├── vizualizacia_tresholds.py       # Výkonnosť pri rôznych prahoch
    ├── rozdiel_k_features.py           # DeLong test pre počet premenných
    └── interpretacia_LogReg.py         # Interpretácia logistickej regresie

│    
└── figures/                            # Výstupné grafy a vizualizácie  
```

---

## Reprodukovateľnosť

Náhodnosť je fixovaná hodnotou `random_state = 42` vo všetkých skriptoch.  
Natrénované modely sú uložené v `models/*.pkl` pred aj po kalibrácií — možno ich použiť priamo bez opakovania tréningu.
