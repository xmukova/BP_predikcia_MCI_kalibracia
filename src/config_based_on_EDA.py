# KONFIGURACNY SUBOR FEATURES, KTORE SU VHODNE PRE MODELOVANIE A FEATURE SELECTION, IDENTIFIKOVANE POCAS EDA
# definicia cielovej premennej
TARGET_COLUMN = "target"

# definicia kategorickych stlpcov identifikovanych pocas EDA
CATEGORICAL_COLUMNS = ['NACCREFR', 'SEX', 'HISPANIC', 'NACCLIVS', 'RESIDENC', 'HANDED', 'NACCFAM', 'NACCMOM',
                        'NACCDAD', 'ANYMEDS', 'TOBAC30', 'TOBAC100', 'CVHATT', 'CVAFIB', 'CVANGIO', 'CVOTHR', 
                        'CBTIA', 'NACCTBI', 'DIABETES', 'HYPERTEN', 'HYPERCHO', 'B12DEF', 'THYROID', 'INCONTU',
                        'ALCOHOL', 'DEP2YRS', 'DEPOTHR', 'PSYCDIS', 'VISION', 'VISCORR', 'HEARING', 'HEARAID',
                        'MEMORY', 'JUDGMENT', 'CDRGLOB', 'NPIQINF', 'AGIT', 'DEPD', 'ANX', 'APA', 'IRR', 'NITE',
                        'APP', 'SATIS', 'DROPACT', 'EMPTY', 'BORED', 'SPIRITS', 'AFRAID', 'HAPPY', 'HELPLESS', 
                        'STAYHOME', 'MEMPROB', 'WONDRFUL', 'ENERGY', 'BETTER', 'TAXES', 'REMDATES', 'TRAVEL', 
                        'NACCNREX', 'DECSUB', 'DECIN', 'COGMEM', 'COGJUDG', 'COGMODE', 'BEDEP', 'BEMODE', 'COGSTAT', 'NORMCOG', 
                        'DEP', 'NACCAAAS', 'NACCAANX', 'NACCAC', 'NACCACEI', 'NACCADEP', 'NACCAHTN', 'NACCANGI', 
                        'NACCBETA', 'NACCCCBS', 'NACCDBMD', 'NACCDIUR', 'NACCEMD', 'NACCHTNC', 'NACCLIPL', 'NACCNSD',
                        'NACCPDMD', 'RACE', 'MARISTAT', 'NACCNINR', 'PACKSPER', 'NACCNIHR', 'NACCCOGF', 'NACCBEHF', 'PRIMLANG']


# definicia kontinualnych stlpcov identifikovanych pocas EDA
CONTINUOUS_COLUMNS = ['EDUC', 'SMOKYRS', 'HEIGHT', 'WEIGHT', 'BPSYS', 'BPDIAS', 'HRATE', 'CDRSUM', 
                      'NACCGDS', 'ANIMALS', 'VEG', 'TRAILA', 'TRAILB', 'NACCAGEB', 'NACCAMD', 'NACCBMI']

FEATURES = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS