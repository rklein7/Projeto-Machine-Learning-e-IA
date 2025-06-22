# Projeto Machine Learning e IA
## Previsão de Evasão de Alunos

![Python](https://img.shields.io/badge/python-4B0082?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

## 📖 Visão Geral do Projeto

Este projeto, desenvolvido para o Nanodegree de Machine Learning & Inteligência Artificial, foca na construção de um modelo de aprendizado de máquina para **identificar alunos com alto risco de evasão** em um curso online da plataforma PensComp. A análise abrange desde o pré-processamento dos dados e análise exploratória até a implementação, avaliação e, crucialmente, a interpretação dos modelos preditivos.

O objetivo principal não é apenas alcançar alta acurácia, mas também **entender os fatores comportamentais** que levam à evasão, utilizando técnicas de Inteligência Artificial Explicável (XAI) como o SHAP para trazer transparência aos resultados.

## 🎯 Problema de Negócio

A evasão de estudantes (churn) é um desafio crítico para plataformas de ensino online, impactando diretamente a receita e a sustentabilidade do negócio. Identificar proativamente os alunos em risco permite que a PensComp tome **ações direcionadas e eficientes**, como:

* Oferecer suporte pedagógico personalizado para alunos com dificuldades.
* Criar campanhas de reengajamento para alunos inativos.
* Ajustar a metodologia de ensino com base nos padrões de comportamento observados.

O projeto busca responder à pergunta: **Com base no comportamento de um aluno na plataforma, podemos classificar seu risco de evasão?**

## 📊 O Dataset

O conjunto de dados contém informações anonimizadas sobre a interação dos alunos com a plataforma. As principais features incluem:

* **Dados de Acesso:** `ts_primeiro_acesso`, `ts_ultimo_acesso`.
* **Dados de Engajamento:** `nr_interacoes_usuario`, `nr_questionarios_finalizados`, `nr_submissoes_codigo`.
* **Dados de Desempenho:** `vl_desempenho_usuario`, `vl_media_notas`.

### Engenharia da Variável-Alvo

A variável-alvo, `evadiu`, foi criada para classificar os alunos. Um aluno é considerado **evadido (1)** se ele satisfaz duas condições de negócio simultaneamente:
1.  Não apresentou nenhum desempenho (`vl_desempenho_usuario = 0`).
2.  Não acessou a plataforma nos últimos 12 dias (`nr_dias_desde_ultimo_acesso <= 12`).

Caso contrário, é considerado um aluno **ativo (0)**.

## 🛠️ Metodologia

O projeto seguiu um fluxo de trabalho estruturado de Ciência de Dados:

1.  **Limpeza e Pré-Processamento:**
    * Remoção de colunas com mais de 70% de dados ausentes.
    * Imputação de valores nulos, utilizando `0` para ausência de atividade (ex: desempenho) e a `mediana` para variáveis contínuas (ex: tempo em questionários), evitando distorções por outliers.
    * Padronização de variáveis textuais e conversão de timestamps.

2.  **Análise Exploratória de Dados (EDA):**
    * Visualização da distribuição das variáveis-chave para entender o perfil dos alunos.
    * Criação de **Perfis de Risco** para segmentar os alunos em categorias acionáveis:
        * `Alto Risco`: Desempenho nulo e inativo.
        * `Reengajamento`: Bom desempenho, mas inativo.
        * `Apoio Pedagógico`: Ativo, mas com baixo desempenho.
        * `Estável`: Ativo e com bom desempenho.

3.  **Modelagem e Avaliação:**
    * Os dados foram divididos em **80% para treino** e **20% para teste**, utilizando amostragem estratificada para manter a proporção da variável-alvo.
    * A performance dos modelos foi validada de forma robusta com **Validação Cruzada Estratificada (`StratifiedKFold` com 5 splits)**.
    * Três modelos de classificação foram treinados e comparados:

| Modelo | Justificativa | Resultado (F1-Score na Validação Cruzada) |
| :--- | :--- | :--- |
| **Regressão Logística** | Simples, interpretável e ótimo como baseline. | `0.8071 ± 0.0850` |
| **Random Forest** | Robusto, lida bem com relações não-lineares. | `0.9599 ± 0.0390` |
| **Rede Neural (Keras)**| Capaz de aprender padrões complexos. | Atingiu `0.96` no teste. |

4.  **Interpretabilidade (XAI com SHAP):**
    * Utilizamos a biblioteca SHAP para "abrir a caixa-preta" dos modelos e entender quais features mais influenciaram suas decisões.

## 📈 Resultados e Conclusões

* **Performance dos Modelos:** Todos os modelos foram eficazes na classificação dos alunos. A Random Forest apresentou F1-score consistente na validação cruzada (≈ 0.93), enquanto a Rede Neural obteve o melhor desempenho no teste real, com excelente equilíbrio entre precisão e recall.
* **Rede Neural como Modelo Final:** A Rede Neural alcançou F1-score de 0.93, Recall de 0.96 e AUC de 0.95 no conjunto de teste real, mostrando ótima capacidade preditiva. Diferente do esperado, ela não apenas aprendeu regras, mas conseguiu generalizar bem para novos dados.
* **Principais Fatores de Evasão:** As variáveis mais relevantes foram vl_desempenho_usuario e nr_dias_desde_ultimo_acesso, validando a regra de negócio e confirmando que baixa participação e desempenho são os maiores preditores de evasão.
* **Insights Acionáveis:** O modelo é útil para estratégias de intervenção pedagógica, permitindo priorizar alunos com alto risco e aumentar a taxa de retenção de forma proativa.

## 🚀 Como Executar o Projeto

Para replicar esta análise, siga os passos abaixo.

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/rklein7/Projeto-Machine-Learning-e-IA.git
    cd Projeto-Machine-Learning-e-IA
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o Jupyter Notebook:**
    Abra o arquivo `analise_exploratoria_e_modelagem.ipynb` em um ambiente Jupyter (Jupyter Lab ou Jupyter Notebook) para explorar a análise completa.
    ```bash
    jupyter-lab
    ```

5.  **Ou execute localmente em uma IDE de sua escolha**    

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Análise de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, TensorFlow/Keras
* **XAI:** SHAP
* **Ambiente:** Jupyter Notebook

## 👨‍🎓 Alunos

* Bruno Pasquetti
* Gabriel Brocco de Oliveira
* Pedro Henrique de Bortoli
* Rafael Augusto Klein
