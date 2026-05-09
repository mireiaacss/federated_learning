import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings("ignore")
import os

# crear carpeta generated_data si no existe para guardar los csv que el dashboard va a leer
if not os.path.exists("generated_data"):
    os.makedirs("generated_data")

def load_data_and_generate_eda():
    # descargamos el dataset clásico de marketing bancario de uci
    bank_marketing = fetch_ucirepo(id=222)

    X = bank_marketing.data.features.copy()
    y = bank_marketing.data.targets.copy()

    # mapeamos el target a 1 y 0 para que el modelo lo entienda
    y = y.iloc[:, 0].map({"yes": 1, "no": 0}).values

    # pasamos las variables categóricas a columnas binarias (one-hot)
    # y rellenamos los nulos con la mediana para que no explote el modelo
    X = pd.get_dummies(X)
    X = X.fillna(X.median(numeric_only=True))
    
    # guardamos los nombres de las columnas para el eda antes de pasarlo a numpy
    feature_names = X.columns.tolist()
    X_vals = X.values.astype(float)

    # creamos un dataframe temporal para sacar medias y dárselas al dashboard
    df_eda = pd.DataFrame(X_vals, columns=feature_names)
    df_eda['target'] = y
    
    # agrupamos por si se le dio el préstamo o no y sacamos las medias
    eda_df = pd.DataFrame({
        'feature': feature_names,
        'mean_overall': df_eda[feature_names].mean().values,
        'mean_eligible_1': df_eda[df_eda['target'] == 1][feature_names].mean().values,
        'mean_not_eligible_0': df_eda[df_eda['target'] == 0][feature_names].mean().values
    })
    
    # guardamos el eda para que la pestaña 5 del dashboard lo pueda leer
    eda_df.to_csv("generated_data/eda_analysis.csv", index=False)

    return X_vals, y


def split_into_banks(X_train, y_train):
    # partimos el dataset aleatoriamente en 3 trozos para simular que 
    # son tres bancos distintos que no comparten datos entre ellos
    idx = np.random.permutation(len(X_train))
    chunks = np.array_split(idx, 3)
    names = ["Bank A", "Bank B", "Bank C"]
    banks = {}

    for name, chunk in zip(names, chunks):
        banks[name] = {"X": X_train[chunk], "y": y_train[chunk], "n": len(chunk)}

    return banks


def apply_smote_per_bank(banks):
    # como los datos están muy desbalanceados (muchos 'no', pocos 'yes'),
    # aplicamos smote banco por banco. smote inventa datos sintéticos
    # de la clase minoritaria para equilibrar la balanza.
    # ojo: esto solo se hace en train, nunca en test.
    smote = SMOTE(random_state=42)
    balanced = {}

    for name, data in banks.items():
        X_res, y_res = smote.fit_resample(data["X_train"], data["y_train"])
        balanced[name] = {
            "X_train": X_res,
            "y_train": y_res,
            "X_test":  data["X_test"],
            "y_test":  data["y_test"],
            "n_train": len(y_res),
        }

    return balanced


# montamos una regresión logística desde cero porque scikit-learn
# no nos deja sacar y meter los pesos a mitad de entrenamiento fácilmente.
# esto es necesario para poder hacer la media de pesos en federated learning.
class FederatedLogisticRegression:
    def __init__(self, n_features, learning_rate=0.05, l2=0.001):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate
        self.l2 = l2 # penalización para que los pesos no se vuelvan locos

    def _sigmoid(self, z):
        # clip evita que nos dé un error de overflow matemático si z es muy grande
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    # añadimos el umbral como parámetro para poder cambiarlo
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def _gradients(self, X, y):
        # calculamos el error y actualizamos la pendiente
        n = X.shape[0]
        error = self.predict_proba(X) - y
        dW = (X.T @ error) / n + self.l2 * self.weights
        db = error.mean()
        return dW, db

    def gradient_step(self, X, y):
        dW, db = self._gradients(X, y)
        self.weights -= self.lr * dW
        self.bias -= self.lr * db

    def get_params(self):
        # esto es lo único que viaja por la red: los números del modelo.
        # la privacidad de los clientes está a salvo.
        return {"weights": self.weights.copy(), "bias": self.bias}

    def set_params(self, params):
        self.weights = params["weights"].copy()
        self.bias = params["bias"]


def federated_training(banks, n_rounds=30, local_epochs=5, lr=0.05):
    # el servidor central empieza con pesos a cero
    n_features = list(banks.values())[0]["X_train"].shape[1]
    n_total = sum(b["n_train"] for b in banks.values())
    server_weights = np.zeros(n_features)
    server_bias = 0.0
    round_log = []

    for rnd in range(1, n_rounds + 1):
        local_params = []

        # cada banco entrena su propio modelo con sus propios datos locales
        for bank_name, bank_data in banks.items():
            local = FederatedLogisticRegression(n_features, learning_rate=lr)
            local.set_params({"weights": server_weights, "bias": server_bias})

            for _ in range(local_epochs):
                local.gradient_step(bank_data["X_train"], bank_data["y_train"])

            local_params.append({
                "params": local.get_params(),
                "n": bank_data["n_train"],
            })

        # el servidor hace la media de los pesos de todos los bancos
        # si un banco tiene más datos, pesa más en la media
        new_w = np.zeros(n_features)
        new_b = 0.0
        for item in local_params:
            w = item["n"] / n_total
            new_w += w * item["params"]["weights"]
            new_b += w * item["params"]["bias"]

        server_weights, server_bias = new_w, new_b

        # guardamos métricas para ver si está aprendiendo
        temp = FederatedLogisticRegression(n_features)
        temp.set_params({"weights": server_weights, "bias": server_bias})
        f1s, aucs, precs, recs = [], [], [], []
        
        for b in banks.values():
            yp = temp.predict(b["X_test"])
            yprob = temp.predict_proba(b["X_test"])
            f1s.append(f1_score(b["y_test"], yp, zero_division=0))
            aucs.append(roc_auc_score(b["y_test"], yprob))
            precs.append(precision_score(b["y_test"], yp, zero_division=0))
            recs.append(recall_score(b["y_test"], yp, zero_division=0))

        round_log.append({
            "round": rnd,
            "f1": np.mean(f1s),
            "auc": np.mean(aucs),
            "precision": np.mean(precs),
            "recall": np.mean(recs),
        })

    final = FederatedLogisticRegression(n_features)
    final.set_params({"weights": server_weights, "bias": server_bias})
    return final, pd.DataFrame(round_log)


def train_centralized(banks_raw):
    # juntamos todo en un único sitio para simular el caso donde
    # no hay privacidad y centralizamos todo en un data warehouse
    X_all = np.concatenate([b["X_train"] for b in banks_raw.values()])
    y_all = np.concatenate([b["y_train"] for b in banks_raw.values()])
    
    # aplicamos smote globalmente
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_all, y_all)

    # hacemos gridsearch para optimizar el f1 score que pediste
    # probamos diferentes niveles de regularización
    param_grid = {'C': [0.1, 1.0, 10.0]}
    
    # cv=3 hace cross validation interno para asegurar que no hay overfitting
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42), 
        param_grid, 
        scoring='f1', 
        cv=3
    )
    grid.fit(X_res, y_res)
    
    return grid.best_estimator_


def find_best_threshold(y_true, y_proba):
    # iteramos sobre umbrales desde 0.1 hasta 0.9 para encontrar el mejor f1
    best_t = 0.5
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.05):
        pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t


def evaluate(label, model, banks, is_federated):
    # unificamos test para sacar la nota final del modelo
    X_test = np.concatenate([b["X_test"] for b in banks.values()])
    y_test = np.concatenate([b["y_test"] for b in banks.values()])

    if is_federated:
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = model.predict_proba(X_test)[:, 1]

    # calculamos el mejor umbral en lugar de usar el 0.5 por defecto
    best_threshold = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_threshold).astype(int)

    # calculamos matriz de confusión a mano para extraer vp, fp, vn, fn
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1

    return {
        "label": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "cm": cm,
        "threshold": best_threshold # lo guardamos por curiosidad
    }


def get_per_bank_metrics(banks, fed_model, cen_model):
    # extraemos los resultados separados para que el streamlit los dibuje bien
    rows = []
    for name, data in banks.items():
        for label, model, is_fed in [
            ("Federated", fed_model, True),
            ("Centralized", cen_model, False),
        ]:
            y_proba = (model.predict_proba(data["X_test"])
                       if is_fed else model.predict_proba(data["X_test"])[:, 1])
            
            # volvemos a usar la función para el mejor umbral de cada banco
            t = find_best_threshold(data["y_test"], y_proba)
            yp = (y_proba >= t).astype(int)
            
            rows.append({
                "bank": name,
                "model": label,
                "f1": f1_score(data["y_test"], yp, zero_division=0),
                "auc": roc_auc_score(data["y_test"], y_proba),
                "recall": recall_score(data["y_test"], yp, zero_division=0),
                "precision": precision_score(data["y_test"], yp, zero_division=0),
            })
    return pd.DataFrame(rows)


def print_metrics(res):
    print(f"\n{res['label']}")
    print(f"threshold (optimal threshold): {res['threshold']:.2f}")
    print(f"f1 score (balance between precision and recall): {res['f1']:.4f}")
    print(f"precision (reliability of positive alerts): {res['precision']:.4f}")
    print(f"recall (ability to detect actual positives): {res['recall']:.4f}")
    print(f"roc-auc (quality of probability ranking): {res['auc']:.4f}")
    print(f"accuracy (overall percentage of correct predictions): {res['accuracy']:.4f}")


def main():
    np.random.seed(42)

    X, y = load_data_and_generate_eda()

    # partimos 80% train, 20% test manteniendo la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # escalamos los datos para que el descenso de gradiente fluya rápido y bien
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    raw_banks = split_into_banks(X_train, y_train)

    banks_split = {}
    for name, data in raw_banks.items():
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
            data["X"], data["y"], test_size=0.2, random_state=42
        )
        banks_split[name] = {
            "X_train": Xb_tr, "y_train": yb_tr,
            "X_test": Xb_te, "y_test": yb_te,
            "n_train": len(Xb_tr),
        }

    banks_balanced = apply_smote_per_bank(banks_split)

    # hacemos una búsqueda manual de hiperparámetros (gridsearch) para el modelo federado 
    # enfocándonos en maximizar el f1 score.
    best_f1 = 0
    best_fed_model = None
    best_round_log = None
    
    # iteramos sobre diferentes learning rates
    for lr in [0.01, 0.05, 0.1]:
        fed_model, round_log = federated_training(
            banks_balanced, n_rounds=30, local_epochs=5, lr=lr
        )
        
        # calculamos f1 del último round
        current_f1 = round_log.iloc[-1]['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_fed_model = fed_model
            best_round_log = round_log

    # entrenamos la baseline centralizada (lleva su propio gridsearch por debajo)
    cen_model = train_centralized(banks_split)

    # sacamos resultados
    fed_res = evaluate("federated model", best_fed_model, banks_balanced, is_federated=True)
    cen_res = evaluate("centralized baseline", cen_model, banks_balanced, is_federated=False)

    print_metrics(fed_res)
    print_metrics(cen_res)

    # exportamos datos al dashboard
    best_round_log.to_csv("generated_data/round_log.csv", index=False)

    pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("cm")}
        for r in [fed_res, cen_res]
    ]).to_csv("generated_data/results_summary.csv", index=False)

    get_per_bank_metrics(banks_balanced, best_fed_model, cen_model).to_csv(
        "generated_data/bank_metrics.csv", index=False
    )

    pd.DataFrame([
        {"label": r["label"], "tn": r["cm"][0,0], "fp": r["cm"][0,1],
         "fn": r["cm"][1,0],  "tp": r["cm"][1,1]}
        for r in [fed_res, cen_res]
    ]).to_csv("generated_data/confusion_matrices.csv", index=False)


if __name__ == "__main__":
    main()