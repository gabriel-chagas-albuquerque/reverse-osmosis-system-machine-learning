# OR-SYSTEM - Previsão de Pressão em Sistema de Osmose Reversa

## 📋 Sobre o Projeto

Este repositório contém o código desenvolvido para o Trabalho de Conclusão de Curso (TCC) de Engenharia Mecânica:

**Título:** PREVISÃO DA PRESSÃO EM UM SISTEMA DE OSMOSE REVERSA COM FONTE EÓLICA A PARTIR DE DADOS EXPERIMENTAIS UTILIZANDO ALGORITMOS DE APRENDIZADO DE MÁQUINA

### Objetivos

Realizar uma análise exploratória e comparativa para:
- Analisar e selecionar a estratégia de pré-processamento de dados mais adequada, comparando o impacto das normalizações Min-Máx e Z-Score no desempenho dos modelos;
- Avaliar e comparar o desempenho preditivo (RMSE e MAE) e a estabilidade (Coeficiente de Variação - CV) de diferentes famílias de algoritmos de ML (MQO, SVR e MLP) para o problema em questão;
- Investigar a sensibilidade dos modelos (em particular SVR e MLP) à variação de seus respectivos hiperparâmetros (como kernels do SVR, e função de ativação/taxa de aprendizado do MLP) para a otimização do desempenho;
- Realizar uma análise da distribuição de erros (via razão RMSE/MAE) para inferir a natureza dos resíduos preditivos e a robustez dos modelos a outliers;
- Validar visualmente a aderência preditiva dos modelos de melhor desempenho frente aos dados experimentais, confirmando sua capacidade de reproduzir a complexidade e a periodicidade da curva de pressão.

### Sistema OR-SYSTEM

O OR-SYSTEM é um sistema de **osmose reversa movido por energia eólica**, composto por:
- **Catavento** (rotor eólico) acionado pela força do vento
- **Bomba de pistão** conectada mecanicamente ao catavento
- **Sistema de osmose reversa** alimentado pela pressão gerada

---

## 🔬 Metodologia

### Variáveis de Entrada
- **VPP** (Velocidade de Ponta de Pá): Calculada a partir da velocidade do vento, rotação da bomba e raio do catavento
- **ang_virab** (Ângulo do Virabrequim): Posição angular do eixo em radianos

### Variável de Saída
- **pressao** (Pressão): Pressão gerada no sistema de osmose reversa (bar)

### Dados Experimentais

Os dados foram obtidos do estudo de **OKURA et al. (2023)**:

> Okura, S. S. et al., 2023. *Evaluation of direct coupling between conventional windmills and reverse osmosis desalination systems at low wind speeds*. Energy Conversion and Management, Volume 295, 1 Novembro.

**Arquivos:**
- `or-system-dados-brutos.csv`: Dados brutos das medições experimentais
- `or-system-database.csv`: Dados pré-processados (um ciclo por configuração, pressões zero removidas)

---

## 🤖 Algoritmos Testados

### Redes Neurais Artificiais (MLP)
- **Funções de ativação**: `tanh`, `logistic` (sigmoid), `relu`
- **Taxas de aprendizado**: 0.005, 0.01
- **Neurônios na camada oculta**: 20 
- **Tamanho dos Minilotes**: 32
- **Algoritmo de Otimização**: Gradiente Descendente Estocástico (SGD)
- **Total**: 6 configurações (3 funções × 2 taxas)

### Support Vector Machines (SVM)
- **Kernels**: `rbf`, `linear`, `poly` (grau 3)
- **Parâmetro epsilon**: 0.1, 0.3
- **Parâmetro C**: 1.0
- **Total**: 6 configurações (3 kernels × 2 epsilon)

### Regressão Linear
- **MQO** (Mínimos Quadrados Ordinários)
- **Ridge** (L2): α = 0.001, 0.01
- **Lasso** (L1): α = 0.0001, 0.001, 0.01
- **Total**: 6 configurações

### Estratégias de Normalização
- **StandardScaler**: Normalização com média zero e desvio padrão unitário
- **MinMaxScaler**: Normalização para intervalo [0, 1]

**TOTAL GERAL:** 36 combinações (18 modelos × 2 normalizações)

---

## 📊 Análises Realizadas

### 1. Visualizações dos Dados

#### Dados Brutos
- Gráficos de pressão vs ângulo do virabrequim para dados originais
- Análise por velocidade do vento (3.5, 4.5, 5.5 m/s)
- Identificação de defasagem entre curvas

#### Dados Tratados
- Gráficos de pressão vs ângulo após pré-processamento
- Análise por velocidade do vento (3.5, 4.5, 5.5 m/s)
- Análise por VPP (0.3073, 0.5145, 0.5274, 0.7652)

#### Dados com Alinhamento (Shift)
- **Correção de defasagem** angular entre curvas
- Gráficos por velocidade do vento com ângulos corrigidos
- Gráficos por VPP com ângulos corrigidos
- Melhoria na comparação entre diferentes rotações

### 2. Comparação de Modelos

#### Ranking Geral (Figura 1)
- Classificação de todos os modelos por RMSE médio
- Comparação lado a lado: StandardScaler vs MinMaxScaler
- Valores de RMSE exibidos para cada modelo

#### Análise de Estabilidade (Figura 3)
- Coeficiente de Variação (CV) de cada modelo
- Apenas para StandardScaler
- Identificação de modelos mais consistentes
- Cores indicativas: verde (estável) a vermelho (instável)

### 3. Análise de Hiperparâmetros

#### MLP (Figura 4)
- Impacto de funções de ativação (tanh, logistic, relu)
- Efeito da taxa de aprendizado (0.005 vs 0.01)
- Apenas para StandardScaler
- Gráficos de barras agrupadas com valores

#### SVM (Figura 5)
- Comparação de kernels (RBF, Linear, Poly)
- Efeito do parâmetro epsilon (0.1 vs 0.3)
- Apenas para StandardScaler
- Gráficos de barras agrupadas com valores

#### Regressão Linear (Figura 6)
- Efeito de regularização Ridge vs Lasso
- Comparação com MQO (sem regularização)
- Diferentes valores de α
- Apenas para StandardScaler
- Cores distintas: MQO (cinza), Ridge (azul), Lasso (vermelho)

### 4. Visualizações de Predição

#### Gráficos de Predição vs Real
- **3 gráficos**: Um para o melhor modelo de cada tipo (MLP, SVM, RL)
- **Apenas StandardScaler**
- Dois subplots por figura:
  - **Scatter Plot**: Correlação entre predito e real + linha ideal (y=x)
  - **Série Temporal**: Comparação sequencial ordenada por valor real

### Métricas de Avaliação
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- Calculadas para conjuntos de **treino** (80%) e **teste** (20%)
- **50 rodadas independentes** de treino e teste
- Estatísticas: média ± desvio padrão

---

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Estrutura de Arquivos

```
or-system/
│
├── or-system.py                    # Script principal
├── or-system-dados-brutos.csv      # Dados experimentais brutos
├── or-system-database.csv          # Dados pré-processados
├── README.md                       # Este arquivo
├── pressao_vs_angulo_vvento_dados_brutos_*.png
├── pressao_vs_angulo_vvento_*.png
├── pressao_vs_angulo_vpps_especificos.png
├── shift_pressao_vs_angulo_vvento_*.png
├── shift_pressao_vs_angulo_vpps_especificos.png
├── figura1_ranking_normalizacoes.png
├── figura3_estabilidade_modelos.png
├── figura4_hiperparametros_mlp.png
├── figura5_hiperparametros_svm.png
├── figura6_regularizacao_rl.png
├── predicao_melhor_MLP_*_Standard.png
├── predicao_melhor_SVM_*_Standard.png
├── predicao_melhor_RL_*_Standard.png
└── tabela_resumo_normalizacoes.csv
```

### Executando o Código

```bash
python or-system.py
```

### Saídas Geradas

O script gera automaticamente:

#### 📈 Gráficos (formato PNG, 600 DPI)

**Visualizações de Dados:**
1. **Dados Brutos** (3 arquivos): Curvas de pressão para cada velocidade de vento (3.5, 4.5, 5.5 m/s)
2. **Dados Tratados** (3 arquivos): Curvas após pré-processamento
3. **Dados Tratados por VPP** (1 arquivo): Comparação de diferentes VPPs
4. **Dados com Shift** (3 arquivos): Curvas após correção de defasagem por velocidade
5. **Dados com Shift por VPP** (1 arquivo): Comparação de VPPs com alinhamento

**Análises Comparativas:**
6. **Figura 1**: Ranking geral de modelos (StandardScaler vs MinMaxScaler)
7. **Figura 3**: Análise de estabilidade (Coeficiente de Variação) - StandardScaler
8. **Figura 4**: Hiperparâmetros MLP (ativação e learning rate) - StandardScaler
9. **Figura 5**: Hiperparâmetros SVM (kernels e epsilon) - StandardScaler
10. **Figura 6**: Regularização RL (Ridge, Lasso, MQO) - StandardScaler

**Predições dos Melhores Modelos:**
11-13. **Gráficos de Predição** (3 arquivos): Melhor MLP, SVM e RL - StandardScaler
   - Subplot 1: Scatter plot (predito vs real)
   - Subplot 2: Série temporal ordenada

#### 📄 Arquivos CSV
- `tabela_resumo_normalizacoes.csv`: Estatísticas completas (média ± desvio padrão) para todos os modelos e normalizadores

---

## 🔧 Configurações Técnicas

### Divisão dos Dados
- **Treino**: 80%
- **Teste**: 20%
- **Seed inicial**: 20000 (incrementado a cada rodada para reprodutibilidade)
- **Número de rodadas**: 50

### Resolução de Imagens
- **DPI**: 600 (qualidade para publicação acadêmica)
- **Formato**: PNG com fundo branco
- **Fonte**: Serif, tamanho 10pt

### Parâmetros dos Modelos

**MLP (MLPRegressor):**
```python
hidden_layer_sizes = (20)           # 20 neurônios na camada oculta
early_stopping = True               # Parada antecipada ativada
max_iter = 1000                     # Máximo de 1000 iterações
n_iter_no_change = 100             # Parar após 100 iterações sem melhoria
tol = 0.001                        # Tolerância para convergência
solver = "sgd"                     # Gradiente Descendente Estocástico
learning_rate = "constant"         # Taxa de aprendizado constante
validation_fraction = 0.2          # 20% para validação
alpha = 0.001                      # Regularização L2
batch_size = 32                    # Tamanho do minilote
```

**SVM (SVR):**
```python
C = 1.0                           # Parâmetro de regularização
tol = 0.001                       # Tolerância para convergência
max_iter = -1                     # Sem limite de iterações
```

**Regressão Linear:**
```python
tol = 0.001                       # Tolerância (Ridge e Lasso)
# MQO não possui hiperparâmetros
```

### Alinhamento de Curvas (Shift)
- **Objetivo**: Remover defasagem angular entre curvas de diferentes rotações
- **Método**: 
  1. Identificar o pico de pressão em cada rotação
  2. Selecionar a rotação com maior pico como referência
  3. Calcular deslocamento angular necessário
  4. Aplicar correção: `ang_virab_corrigido = ang_virab + shift`
- **Aplicado apenas para**: Velocidades de vento 3.5, 4.5, 5.5 m/s

---

## 📊 Interpretação dos Resultados

### Coeficiente de Variação (CV)
- **CV < 5%**: Modelo muito estável e consistente
- **CV 5-10%**: Estabilidade moderada
- **CV > 10%**: Alta variabilidade entre rodadas

### RMSE (Root Mean Squared Error)
- Penaliza fortemente erros grandes
- Valores mais baixos indicam melhor desempenho
- Unidade: bar (mesma da pressão)

### MAE (Mean Absolute Error)
- Menos sensível a outliers que RMSE
- Interpretação mais direta: erro médio absoluto
- Unidade: bar

### Análise de Hiperparâmetros

**MLP - Funções de Ativação:**
- **tanh**: Geralmente mais estável, saída em [-1, 1]
- **logistic**: Similar a tanh, saída em [0, 1]
- **relu**: Pode ser mais rápida, mas menos estável em alguns casos

**MLP - Taxa de Aprendizado:**
- **0.01**: Convergência mais rápida, risco de instabilidade
- **0.005**: Convergência mais lenta, geralmente mais estável

**SVM - Kernels:**
- **RBF**: Não-linear, flexível
- **Linear**: Mais simples, útil para relações lineares
- **Poly**: Não-linear polinomial, pode capturar interações complexas

**SVM - Epsilon:**
- Controla a largura da "zona de indiferença"
- Valores maiores: modelo mais tolerante a erros

**RL - Regularização:**
- **Ridge (L2)**: Reduz magnitude dos coeficientes
- **Lasso (L1)**: Pode zerar coeficientes (seleção de features)
- **MQO**: Sem regularização, pode sofrer de overfitting

---

## 🔧 Personalização

### Modificar Hiperparâmetros

Edite as listas de configuração no código:

```python
mlp_configs = [
    {'name': 'MLP1', 'activation': 'tanh', 'learning_rate': 0.01, 'verbose': False},
    # Adicione mais configurações aqui
]

svm_configs = [
    {'name': 'SVM1', 'kernel': 'rbf', 'C': 1, 'epsilon': 0.1},
    # Adicione mais configurações aqui
]

rl_configs = [
    {'name': 'RL1', 'penalty': 'l2', 'alpha': 0.001},
    # Adicione mais configurações aqui
]
```

### Alterar Número de Rodadas

```python
n_rodadas = 50  # Altere este valor
```

**Nota:** Mais rodadas = estatísticas mais robustas, mas maior tempo de execução.

### Modificar Normalização

Para adicionar ou remover normalizadores:

```python
normalizadores = [
    {'nome': 'MinMax', 'scaler_class': MinMaxScaler},
    {'nome': 'Standard', 'scaler_class': StandardScaler},
    # Adicione outros scalers do scikit-learn se desejar
]
```

---

## ⚠️ Observações Importantes

### Foco em StandardScaler

Por decisão de projeto, as análises de hiperparâmetros e predição focam **apenas no StandardScaler**:
- **Figura 3** (Estabilidade): Apenas StandardScaler
- **Figura 4** (MLP): Apenas StandardScaler
- **Figura 5** (SVM): Apenas StandardScaler
- **Figura 6** (RL): Apenas StandardScaler
- **Gráficos de Predição**: Apenas StandardScaler

Esta escolha simplifica a análise e evita redundância visual.

### Valores Exibidos nos Gráficos

Todos os gráficos de barras exibem os valores numéricos **fora das barras** para melhor legibilidade, especialmente em impressões P&B.

### Formato Brasileiro

Todos os números são formatados no padrão brasileiro:
- Vírgula como separador decimal (ex: 1,234)
- Ponto como separador de milhares (ex: 1.000)

---

## 📚 Referências

**Dados Experimentais:**
> Okura, S. S. et al., 2023. Evaluation of direct coupling between conventional windmills and reverse osmosis desalination systems at low wind speeds. *Energy Conversion and Management*, Volume 295, 1 Novembro.

**Bibliotecas Utilizadas:**
- [scikit-learn](https://scikit-learn.org/): Algoritmos de ML
- [pandas](https://pandas.pydata.org/): Manipulação de dados
- [NumPy](https://numpy.org/): Computação numérica
- [Matplotlib](https://matplotlib.org/): Visualização

---

## 👨‍🎓 Autor

Trabalho de Conclusão de Curso  
**Autor:** Gabriel das Chagas Albuquerque
**Orientador:** Prof. Dr. Francisco Frederico dos Santos Matos  
**Curso:** Engenharia Mecânica  
**Ano:** 2025

---


## 🤝 Contribuições

Este é um projeto acadêmico (TCC). Sugestões e melhorias são bem-vindas através de issues ou pull requests.

---

## 📧 Contato

Para dúvidas sobre o projeto, entre em contato através do repositório ou pelo e-mail gabriel.chagas.albuquerque08@ifce.edu.br

---

**⚡ Desenvolvido com Python e Scikit-learn para análise de sistemas de osmose reversa eólicos**