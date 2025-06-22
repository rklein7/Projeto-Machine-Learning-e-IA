from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API de Previsão de Evasão")

# Carregar modelo e scaler
model = joblib.load("modelo_rf.pkl")
scaler = joblib.load("scaler.pkl")

# Classe com todos os campos esperados
class DadosEntrada(BaseModel):
    nr_dias_desde_primeiro_acesso: float
    nr_atividades_sinalizadas: float
    nr_atividades_mapeadas: float
    nr_discussoes_forum_postadas: float
    nr_questionarios_abandonados: float
    nr_questionarios_finalizados: float
    vl_medio_tempo_questionario: float
    vl_medio_tempo_questionario_avaliado: float
    vl_desempenho_questionario: float
    nr_intervalos_uso: float
    nr_dias_uso: float
    vl_medio_atividade_diaria: float
    vl_engajamento_usuario_por_intervalo: float
    vl_engajamento_usuario_intradia: float
    nr_interacoes_usuario: float
    nr_dias_engajamento_discussao: float
    nr_dias_engajamento_questionario: float
    nr_engajamento_discussao: float
    nr_engajamento_questionario: float
    nr_questoes_respondidas: float
    nr_questoes_corretas: float
    nr_questoes_erradas: float
    nr_questoes_parciais: float
    vl_desempenho_usuario: float
    nr_itens_avaliados: float
    nr_itens_nao_respondidos: float
    vl_media_notas: float
    nr_submissoes_codigo: float


@app.post("/prever/")
def prever(dados: DadosEntrada):
    df = pd.DataFrame([dados.dict()])
    df_scaled = scaler.transform(df)
    proba = model.predict_proba(df_scaled)[0][1]
    limiar = 0.7
    pred = int(proba > limiar)
    return {
        "evadiu": bool(pred),
        "probabilidade": round(proba, 4)
    }

