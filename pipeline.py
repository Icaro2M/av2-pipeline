import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_log_error


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        self.global_mean_ = y.mean()
        self.maps_ = {}

        for col in X.columns:
            temp = pd.DataFrame({
                "category": X[col].astype(str),
                "target": y
            })

            stats = temp.groupby("category")["target"].agg(["mean", "count"])

            smooth_mean = (
                stats["mean"] * stats["count"] +
                self.global_mean_ * self.smoothing
            ) / (stats["count"] + self.smoothing)

            self.maps_[col] = smooth_mean.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().reset_index(drop=True)

        encoded = pd.DataFrame(index=X.index)

        for col in X.columns:
            encoded[f"{col}_target_enc"] = (
                X[col]
                .astype(str)
                .map(self.maps_[col])
                .fillna(self.global_mean_)
            )

        return encoded


sys.modules["__main__"].TargetMeanEncoder = TargetMeanEncoder


def tratar_dados_inicial(df):
    df = df.copy()

    cols_none = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
    ]

    cols_none_existentes = [col for col in cols_none if col in df.columns]
    df[cols_none_existentes] = df[cols_none_existentes].fillna("None")

    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    if "MasVnrType" in df.columns:
        df["MasVnrType"] = df["MasVnrType"].fillna("None")

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    if "Electrical" in df.columns and df["Electrical"].isna().any():
        moda = df["Electrical"].mode()
        if len(moda) > 0:
            df["Electrical"] = df["Electrical"].fillna(moda[0])

    return df


def criar_features_base(df):
    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"] +
        df["1stFlrSF"] +
        df["2ndFlrSF"]
    )

    df["TotalBathrooms"] = (
        df["FullBath"] +
        0.5 * df["HalfBath"] +
        df["BsmtFullBath"] +
        0.5 * df["BsmtHalfBath"]
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df.loc[df["GarageYrBlt"] == 0, "GarageAge"] = 0

    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFence"] = (df["Fence"] != "None").astype(int)

    df["HasPorch"] = (
        df["OpenPorchSF"] +
        df["EnclosedPorch"] +
        df["3SsnPorch"] +
        df["ScreenPorch"] > 0
    ).astype(int)

    df["TotalPorchSF"] = (
        df["OpenPorchSF"] +
        df["EnclosedPorch"] +
        df["3SsnPorch"] +
        df["ScreenPorch"]
    )

    df["HouseAge"] = df["HouseAge"].clip(lower=0)
    df["RemodAge"] = df["RemodAge"].clip(lower=0)
    df["GarageAge"] = df["GarageAge"].clip(lower=0)

    return df


def criar_features_fe3_publico(df):
    df = df.copy()

    high_neighborhoods = ["NridgHt", "NoRidge", "StoneBr"]
    mid_high_neighborhoods = [
        "Veenker",
        "Somerst",
        "Timber",
        "ClearCr",
        "Crawfor",
        "CollgCr",
    ]

    df["NeighborhoodTier"] = "Low"
    df.loc[df["Neighborhood"].isin(mid_high_neighborhoods), "NeighborhoodTier"] = "MidHigh"
    df.loc[df["Neighborhood"].isin(high_neighborhoods), "NeighborhoodTier"] = "High"

    tier_map = {
        "Low": 0,
        "MidHigh": 1,
        "High": 2,
    }

    df["NeighborhoodTierNum"] = df["NeighborhoodTier"].map(tier_map)

    area_cols = [
        "LotArea",
        "GrLivArea",
        "TotalSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "TotalPorchSF",
    ]

    for col in area_cols:
        df[f"log_{col}"] = np.log1p(df[col])

    df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["OverallQual_TotalBathrooms"] = df["OverallQual"] * df["TotalBathrooms"]
    df["OverallQual_GarageCars"] = df["OverallQual"] * df["GarageCars"]

    df["NeighborhoodTier_TotalSF"] = df["NeighborhoodTierNum"] * df["TotalSF"]
    df["NeighborhoodTier_OverallQual"] = df["NeighborhoodTierNum"] * df["OverallQual"]

    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasWoodDeck"] = (df["WoodDeckSF"] > 0).astype(int)
    df["HasOpenPorch"] = (df["OpenPorchSF"] > 0).astype(int)
    df["HasMasVnr"] = (df["MasVnrArea"] > 0).astype(int)
    df["HasMiscFeature"] = (df["MiscFeature"] != "None").astype(int)

    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["RecentRemodel"] = (df["RemodAge"] <= 10).astype(int)
    df["NewHouse"] = (df["HouseAge"] <= 5).astype(int)
    df["OldHouse"] = (df["HouseAge"] >= 80).astype(int)

    return df


def aplicar_correcao_baixas_publico(pred, features, config):
    pred_corrigida = pd.Series(pred).reset_index(drop=True)
    features = features.reset_index(drop=True)

    mascara = (
        (features["OldHouse"] == 1) &
        (features["OverallQual"] <= 4) &
        (features["GrLivArea"] <= 1200) &
        (pred_corrigida >= config["limite_inferior"]) &
        (pred_corrigida < config["limite_superior"])
    )

    pred_corrigida.loc[mascara.values] *= config["fator"]
    pred_corrigida = np.maximum(pred_corrigida, 0)

    return pred_corrigida.values


def aplicar_correcao_altas_publico(pred, features, config):
    pred_corrigida = pd.Series(pred).reset_index(drop=True)
    features_temp = features.reset_index(drop=True).copy()
    features_temp["pred_base"] = pred_corrigida.values

    regra = config["regra"]

    if regra == "partial_qual_alta_sem_mssub60":
        mascara_regra = (
            (features_temp["SaleCondition"] == "Partial") &
            (features_temp["OverallQual"] >= 8) &
            (features_temp["MSSubClass"] != 60)
        )
    else:
        return pred_corrigida.values

    mascara = (
        mascara_regra &
        (features_temp["pred_base"] >= config["limite_inferior"]) &
        (features_temp["pred_base"] < config["limite_superior"])
    )

    pred_corrigida.loc[mascara.values] *= config["fator"]
    pred_corrigida = np.maximum(pred_corrigida, 0)

    return pred_corrigida.values


def aplicar_correcoes_finais_publico(pred, features, correcao_baixas_config, correcao_altas_config):
    pred_corrigida = aplicar_correcao_baixas_publico(
        pred,
        features,
        correcao_baixas_config
    )

    pred_corrigida = aplicar_correcao_altas_publico(
        pred_corrigida,
        features,
        correcao_altas_config
    )

    return pred_corrigida


def prever_precos(caminho_arquivo_teste):
    """
    Função obrigatória para o corretor automático.
    Lê o arquivo de teste, aplica o pipeline final e retorna as predições.

    Parâmetros:
    caminho_arquivo_teste (str): Caminho local para o arquivo CSV de teste.

    Retorna:
    np.array: As predições de preços (não negativas).
    """
    df_teste_raw = pd.read_csv(caminho_arquivo_teste)

    caminho_modelo = "modelo_final_house_prices.joblib"
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"O arquivo do modelo '{caminho_modelo}' não foi encontrado na raiz do projeto."
        )

    pacote_modelo_final = joblib.load(caminho_modelo)

    modelos_sklearn_finais = pacote_modelo_final["modelos_sklearn"]
    xgb_final = pacote_modelo_final["xgb_final"]
    preprocessor_xgb_final = pacote_modelo_final["preprocessor_xgb_final"]

    pesos_finais_corrigidos = pacote_modelo_final["pesos"]
    colunas_por_modelo = pacote_modelo_final["colunas_por_modelo"]

    correcao_baixas_config = pacote_modelo_final["correcao_baixas"]
    correcao_altas_config = pacote_modelo_final["correcao_altas"]

    df_features = df_teste_raw.copy()

    if "SalePrice" in df_features.columns:
        df_features = df_features.drop(columns=["SalePrice"])

    teste_tratado = tratar_dados_inicial(df_features)
    teste_base = criar_features_base(teste_tratado)
    teste_fe3 = criar_features_fe3_publico(teste_base)

    X_test_base = teste_base.drop(columns=["Id"], errors="ignore")
    X_test_fe3 = teste_fe3.drop(columns=["Id"], errors="ignore")

    predicoes_teste = {}

    for nome_modelo, modelo in modelos_sklearn_finais.items():
        if nome_modelo == "GradientBoosting_base":
            X_test = X_test_base.copy()
        else:
            X_test = X_test_fe3.copy()

        colunas_treino = colunas_por_modelo[nome_modelo]
        colunas_faltando = set(colunas_treino) - set(X_test.columns)

        if colunas_faltando:
            raise ValueError(
                f"{nome_modelo}: colunas faltando no teste: {sorted(colunas_faltando)}"
            )

        X_test = X_test[colunas_treino]

        pred = modelo.predict(X_test)
        pred = np.maximum(pred, 0)

        predicoes_teste[nome_modelo] = pred

    colunas_xgb = colunas_por_modelo["XGBoost_fe3"]
    colunas_faltando_xgb = set(colunas_xgb) - set(X_test_fe3.columns)

    if colunas_faltando_xgb:
        raise ValueError(
            f"XGBoost_fe3: colunas faltando no teste: {sorted(colunas_faltando_xgb)}"
        )

    X_test_xgb = X_test_fe3[colunas_xgb]
    X_test_xgb_proc = preprocessor_xgb_final.transform(X_test_xgb)

    pred_xgb = np.expm1(xgb_final.predict(X_test_xgb_proc))
    pred_xgb = np.maximum(pred_xgb, 0)

    predicoes_teste["XGBoost_fe3"] = pred_xgb

    pred_ensemble = np.zeros(len(df_teste_raw))

    for nome_modelo, peso in pesos_finais_corrigidos.items():
        if nome_modelo not in predicoes_teste:
            raise ValueError(f"Predição do modelo '{nome_modelo}' não foi encontrada.")

        pred_ensemble += peso * predicoes_teste[nome_modelo]

    pred_ensemble = np.maximum(pred_ensemble, 0)

    predicoes_finais = aplicar_correcoes_finais_publico(
        pred_ensemble,
        teste_fe3,
        correcao_baixas_config,
        correcao_altas_config
    )

    predicoes_finais = np.clip(predicoes_finais, a_min=0, a_max=None)

    return predicoes_finais


if __name__ == "__main__":
    arquivo_teste_exemplo = "../../treino.csv"

    print("--- Executando Validação Local do Pipeline ---")

    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' não encontrado.")
        print("Dica: coloque o arquivo de teste na raiz do projeto para testar o script.")
    else:
        try:
            resultados = prever_precos(arquivo_teste_exemplo)

            print("\n✅ Sucesso! O pipeline rodou corretamente.")
            print("-" * 30)
            print("Primeiras 5 predições:")
            print(resultados[:5])
            print("-" * 30)

            df_val = pd.read_csv(arquivo_teste_exemplo)
            if "SalePrice" in df_val.columns:
                y_true = df_val["SalePrice"]
                rmsle = np.sqrt(mean_squared_log_error(y_true, resultados))
                print(f"Métrica RMSLE Local: {rmsle:.5f}")
            else:
                print("[Nota] Coluna 'SalePrice' não encontrada no CSV. Cálculo do RMSLE pulado.")

        except Exception as e:
            print("\n❌ Erro encontrado no pipeline:")
            print(str(e))
            print("\nVerifique se o modelo, o CSV e as dependências estão corretos.")