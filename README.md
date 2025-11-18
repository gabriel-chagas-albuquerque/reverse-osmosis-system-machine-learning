# OR-SYSTEM - Previsão de Pressão em Sistema de Osmose Reversa

## 📋 Sobre o Projeto

Este repositório contém o código desenvolvido para o Trabalho de Conclusão de Curso (TCC) de Engenharia Mecânica:

**Título:** ALGORITMOS DE APRENDIZADO DE MÁQUINA PARA A PREVISÃO DA PRESSÃO EM UM SISTEMA DE OSMOSE REVERSA A PARTIR DE DADOS EXPERIMENTAIS

### Objetivo

Realizar uma análise exploratória e comparativa para:
- Avaliar a **não linearidade** do problema de previsão de pressão
- Investigar a **viabilidade de técnicas de Machine Learning** para predição
- Comparar diferentes **algoritmos** e **estratégias de normalização**
- Analisar o impacto de **hiperparâmetros** no desempenho dos modelos

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
- **Funções de ativação**: `tanh`, `sigmoid`, `relu`
- **Taxas de aprendizado**: 0.005, 0.01
- Configuração: 20 neurônios na camada oculta

### Support Vector Machines (SVM)
- **Kernels**: `rbf`, `linear`, `poly` (grau 3)
- **Parâmetro epsilon**: 0.1, 0.3
- **Parâmetro C**: 1.0

### Regressão Linear
- **MQO** (Mínimos Quadrados Ordinários)
- **Ridge** (L2): α = 0.1, 10
- **Lasso** (L1): α = 0.1, 0.5, 1

### Estratégias de Normalização
- **StandardScaler**: Padronização com média zero e desvio padrão unitário
- **MinMaxScaler**: Escalonamento para intervalo [0, 1]

**Total:** 36 combinações (18 modelos × 2 normalizações)

---

## 📊 Análises Realizadas

### 1. Visualizações dos Dados
- Gráficos de pressão vs ângulo do virabrequim (dados brutos e tratados)
- Análise por velocidade do vento (3.5, 4.5, 5.5 m/s)
- Análise por VPP (0.3073, 0.5145, 0.5274, 0.7652)
- **Correção de defasagem** (shift) entre curvas

### 2. Comparação de Modelos
- **Ranking geral** por normalização (Figura 1)
- **Impacto da normalização** - % de melhoria (Figura 2)
- **Análise de estabilidade** - Coeficiente de Variação (Figura 3)

### 3. Análise de Hiperparâmetros
- **MLP**: Impacto de funções de ativação e learning rate (Figura 4)
- **SVM**: Comparação de kernels e epsilon (Figura 5)
- **Regressão Linear**: Efeito de regularização Ridge/Lasso (Figura 6)

### Métricas de Avaliação
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- Calculadas para conjuntos de **treino** (80%) e **teste** (20%)
- 50 rodadas independentes de treino e teste

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
│
└── outputs/                        # Gerados automaticamente
    ├── pressao_vs_angulo_*.png
    ├── shift_pressao_vs_angulo_*.png
    ├── figura1_ranking_normalizacoes.png
    ├── figura2_impacto_normalizacao.png
    ├── figura3_estabilidade_modelos.png
    ├── figura4_hiperparametros_mlp.png
    ├── figura5_hiperparametros_svm.png
    ├── figura6_regularizacao_rl.png
    └── tabela_resumo_normalizacoes.csv
```

### Executando o Código

```bash
python or-system.py
```

### Saídas Geradas

O script gera automaticamente:

#### 📈 Gráficos (formato PNG, 600 DPI)
1. **Dados Brutos**: Curvas de pressão para cada velocidade de vento
2. **Dados Tratados**: Curvas após pré-processamento
3. **Dados com Shift**: Curvas após correção de defasagem
4. **Figura 1**: Ranking geral de modelos por normalização
5. **Figura 2**: Impacto percentual da normalização
6. **Figura 3**: Análise de estabilidade (CV)
7. **Figura 4**: Hiperparâmetros MLP
8. **Figura 5**: Hiperparâmetros SVM
9. **Figura 6**: Regularização RL

#### 📄 Arquivos CSV
- `tabela_resumo_normalizacoes.csv`: Estatísticas completas (média ± desvio padrão)

---

## 📐 Configurações Técnicas

### Divisão dos Dados
- **Treino**: 80%
- **Teste**: 20%
- **Seed**: 20000 (reprodutibilidade)

### Resolução de Imagens
- **DPI**: 600 (qualidade para publicação acadêmica)
- **Formato**: PNG com fundo branco
- **Fonte**: Serif, tamanho 10pt

### Parâmetros dos Modelos

**MLP:**
```python
hidden_layer_sizes=(20)
early_stopping=True
max_iter=1000
tol=0.001
solver="sgd"
batch_size=32
```

**SVM:**
```python
C=1.0
tol=0.001
max_iter=-1  # Sem limite
```

**Regressão Linear:**
```python
tol=0.001
```

---

## 📊 Interpretação dos Resultados

### Coeficiente de Variação (CV)
- **CV < 5%**: Modelo muito estável
- **CV 5-10%**: Estabilidade moderada
- **CV > 10%**: Alta variabilidade

### Comparação de Normalização
- **Valores negativos**: MinMaxScaler melhor que StandardScaler
- **Valores positivos**: StandardScaler melhor que MinMaxScaler

---

## 🔧 Personalização

### Modificar Hiperparâmetros

Edite as listas de configuração no código:

```python
mlp_configs = [
    {'name': 'MLP_tanh_lr_0.01', 
     'activation': 'tanh',
     'learning_rate': 0.01, 
     'verbose': False},
    # Adicione mais configurações aqui
]
```

### Alterar Número de Rodadas

```python
n_rodadas = 50  
```

**Nota:** Mais rodadas = estatísticas mais robustas, mas maior tempo de execução.


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
**Curso:** Engenharia Mecânica  
**Ano:** 2025

---


---

## 🤝 Contribuições

Este é um projeto acadêmico (TCC). Sugestões e melhorias são bem-vindas através de issues ou pull requests.

---

## 📧 Contato

Para dúvidas sobre o projeto, entre em contato através do repositório ou com o orientador do TCC.

---

**⚡ Desenvolvido com Python e Scikit-learn para análise de sistemas de osmose reversa eólicos**
