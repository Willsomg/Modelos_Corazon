import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, url_for
import pandas as pd
import mlflow.sklearn
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configura MLflow y carga el modelo
mlflow.set_tracking_uri("http://localhost:9090")
MODEL_URI = "models:/corazon/1"
model = mlflow.sklearn.load_model(MODEL_URI)

# Campos numéricos del modelo
NUM_FIELDS = [
    "Gender", "CK-MB", "Troponin", "Age",
    "Heart rate", "Systolic blood pressure",
    "Diastolic blood pressure", "Blood sugar"
]
RISK_LABEL = {0: "Sin riesgo", 1: "Con riesgo"}

# Directorio y archivo de registros
DATA_DIR = 'Data'
RECORDS_FILE = os.path.join(DATA_DIR, 'records.csv')

# Asegurar que existe la carpeta Data/
os.makedirs(DATA_DIR, exist_ok=True)

# Crear el CSV con cabecera si no existe
if not os.path.exists(RECORDS_FILE):
    pd.DataFrame(columns=NUM_FIELDS + ["Prediction", "Timestamp"])\
      .to_csv(RECORDS_FILE, index=False)

# Preparar el StandardScaler igual que en entrenamiento
df_full = pd.read_csv('Data/medical_data.csv')
scaler = StandardScaler().fit(df_full[NUM_FIELDS])


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None

    if request.method == "POST":
        try:
            record = {}
            for f in NUM_FIELDS:
                val = request.form.get(f)
                if not val:
                    raise ValueError(f"Falta el campo '{f}'")
                record[f] = float(val.replace(',', '.').strip())

            df_new = pd.DataFrame([record])
            df_new[NUM_FIELDS] = scaler.transform(df_new[NUM_FIELDS])
            df_aligned = df_new.reindex(
                columns=model.feature_names_in_, fill_value=0
            )

            code = int(model.predict(df_aligned)[0])
            prediction = RISK_LABEL[code]
            timestamp = datetime.now().isoformat()

            # Guardar en Data/records.csv
            row = {**record, "Prediction": prediction, "Timestamp": timestamp}
            pd.DataFrame([row]).to_csv(
                RECORDS_FILE, mode='a', header=False, index=False
            )

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        fields=NUM_FIELDS,
        prediction=prediction,
        error=error
    )


@app.route("/registros")
def registros():
    df = pd.read_csv(RECORDS_FILE)
    if df.empty:
        return render_template("registros.html", no_data=True)

    total_count = len(df)
    df_last = df.tail(5)

    # Gráfico Total de casos con verde pastel y rojo terracota
    counts_risk = df["Prediction"].value_counts()
    plt.figure(figsize=(6,4))
    pastel_colors = [
        '#7FC97F' if pred == 'Sin riesgo' else '#D9534F'
        for pred in counts_risk.index
    ]
    plt.bar(counts_risk.index, counts_risk.values, color=pastel_colors)
    plt.title("Total de casos")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig("static/riesgo.png")
    plt.close()

    return render_template(
        "registros.html",
        no_data=False,
        total_count=total_count,
        table=df_last.rename(columns={
            'Gender': 'Género',
            'CK-MB': 'CK-MB',
            'Troponin': 'Troponina',
            'Age': 'Edad',
            'Heart rate': 'Frecuencia cardíaca',
            'Systolic blood pressure': 'Presión sistólica',
            'Diastolic blood pressure': 'Presión diastólica',
            'Blood sugar': 'Glucosa en sangre',
            'Prediction': 'Predicción',
            'Timestamp': 'Fecha/Hora'
        }).to_html(classes="data", index=False),
        img_riesgo="riesgo.png"
    )


if __name__ == "__main__":
    app.run(debug=True)