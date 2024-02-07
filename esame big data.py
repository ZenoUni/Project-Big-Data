import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


def drop_database(db):
    db.drop(
        ["company_size", "company_location", "work_setting", "employment_type", "employee_residence", "salary_in_usd"],
        axis=1, inplace=True)

def transform_str_int_cat(db):
    count = 0
    for i in range(len(db)):
        if (db.iloc[count, 2] == "Data Analysis"):
            db.iloc[count, 2] = 0
        if (db.iloc[count, 2] == "Data Science and Research"):
            db.iloc[count, 2] = 1
        if (db.iloc[count, 2] == "Data Engineering"):
            db.iloc[count, 2] = 2
        if (db.iloc[count, 2] == "Machine Learning and AI"):
            db.iloc[count, 2] = 3
        if (db.iloc[count, 2] == "Leadership and Management"):
            db.iloc[count, 2] = 4
        if (db.iloc[count, 2] == "BI and Visualization"):
            db.iloc[count, 2] = 5
        if (db.iloc[count, 2] == "Data Quality and Operations"):
            db.iloc[count, 2] = 6
        count += 1

def salari_medi_categorie(db1, db2, categorie1, categorie2):
    y1s = []
    y2s = []
    #MEMORIZZO LA MEDIA SALARIALE DELLE DIVERSE CATEGORIE IN y1s
    for i in range(len(categorie1)):
        y = db1[db1["job_category"].isin([categorie1[i]])]
        y1s.append((y["salary"].mean()))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.bar(categorie1, y1s, color="green")
    ax.set_title('Salario medio delle categorie lavorative 2020')
    ax.set_xlabel('categorie 2020')
    ax.set_ylabel('salario medio 2020')
    # MEMORIZZO LA MEDIA SALARIALE DELLE DIVERSE CATEGORIE IN y2s
    for j in range(len(categorie2)):
        y = db2[db2["job_category"].isin([categorie2[j]])]
        y2s.append((y["salary"].mean()))
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.bar(categorie2, y2s, color="red")
    ax.set_title('Salario medio delle categorie lavorative 2023')
    ax.set_xlabel('categorie 2023')
    ax.set_ylabel('salario medio 2023')
    plt.tight_layout()
    plt.show()

def salari_medi_lavori(db1, db2, lavori1, lavori2):
    y1s = []
    y2s = []
    for i in range(len(lavori1)):
        y = db1[db1["job_title"].isin([lavori1[i]])]
        y1s.append((y["salary"].mean()))
    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(111)
    ax.bar(lavori1, y1s, color="green")
    ax.set_title('Salario medio dei lavori 2020')
    ax.set_xlabel('lavori 2020')
    ax.set_ylabel('salario medio 2020')
    for j in range(len(lavori2)):
        y = db2[db2["job_title"].isin([lavori2[j]])]
        y2s.append((y["salary"].mean()))
    # TRAMITE IL COMANDO SOTTOSTANTE HO AUMENTATO NOTEVOLMENTE
    # LA LARGHEZZA DEL GRAFICO PER SEPARARE BENE I NOMI DEI LAVORI
    fig = plt.figure(figsize=(100, 20))
    ax = fig.add_subplot(111)
    ax.bar(lavori2, y2s, color="red")
    ax.set_title('Salario medio dei lavori 2023')
    ax.set_xlabel('lavori 2023')
    ax.set_ylabel('salario medio 2023')
    plt.tight_layout()
    plt.show()

def regressione_logistica(X_train, X_test, y_train, y_test):
    # NORMALIZZO LE FEATURES
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # CREO E ADDESTRO IL MODELLO DELLA REGRESSIONE LOGISTICA
    logistic_clf = LogisticRegression(random_state=42)
    logistic_clf.fit(X_train, y_train)

    # EFFETTUO LE PREVISIONI
    y_pred = logistic_clf.predict(X_test)

    # CALCOLO L'ACCURACY DEI RISULTATI
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # CALCOLO LA CONFUSION MATRIX E LA VISUALIZZO
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

def legenda_cm():
    print("nella Confusion Matrix: ")
    print("il valore 0 = Entry_level")
    print("il valore 1 = Executive")
    print("il valore 2 = Mid_level")
    print("il valore 3 = Senior")

# LEGGIAMO IL DATASET E LO INSERIAMO NELLA VARIABILE
df = pd.read_csv('jobs_in_data.csv')

# DROPPIAMO LE COLONNE CHE NON CI SERVIRANNO
drop_database(df)

# SELEZIONIAMO SOLO I DATI CON LA VALUTA IN EURO
df = df[df["salary_currency"].isin(["EUR"])]

# CREIAMO 2 DATASET CON I DATI REGISTRATI SOLO NEL 2023 E NEL 2020
df2023 = df[df["work_year"].isin([2023])]
df2020 = df[df["work_year"].isin([2020])]
print(df2023["experience_level"].unique())

# MEMORIZZO LE CATEGORIE DI LAVORO DEI 2 DATASETS IN 2 VARIABILI
# E CHIAMO LA FUNZIONE CHE MI STAMPERA' I GRAFICI
categorie_2020 = df2020["job_category"].unique()
categorie_2023 = df2023["job_category"].unique()
salari_medi_categorie(df2020, df2023, categorie_2020, categorie_2023)

# FACCIO LA STESSA COSA, MA CON I LAVORI
lavori_2020 = df2020["job_title"].unique()
lavori_2023 = df2023["job_title"].unique()
salari_medi_lavori(df2020, df2023, lavori_2020, lavori_2023)

# TRASFORMO LE STRINGHE IN VALORI NUMERICI IN BASE ALLA CATEGORIA DI LAVORO
transform_str_int_cat(df2023)
print("i codici per le categorie sono: ")
for i in range(len(categorie_2023)):
    print("il codice %d sta' per la categoria %s" % (df2023["job_category"].unique()[i], categorie_2023[i]))

# DIVIDO IL DATASET PER VALORI DI PARTENZA E VALORI DA OTTENERE
x = df2023.loc[:, ["job_category", "salary"]]
y = df2023.loc[:, ["experience_level"]]

# DIVIDO I DATI IN TRAIN E TEST E CHOIAMO LA MIA FUNZIONE PER LA REGRESSIONE
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
regressione_logistica(X_train, X_test, y_train, y_test)
# STAMPO NEL TERMINALE IL SIGNIFICATO DEI VALORI DELLA CONFUSION MATRIX
legenda_cm()

