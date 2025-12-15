# ============================================
# OR-SYSTEM - MACHINE LEARNING PARA TCC
# COM PRINTS DETALHADOS E TABELAS VISUAIS
# AUTOR: GABRIEL DAS CHAGAS ALBUQUERQUE
# DATA: DEZEMBRO DE 2025
# VERSÃO: 1.0
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração visual
plt.rcParams.update({
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    'font.family': 'serif',
    'font.size': 10,
})

# Definir estilos de linha para variação
ESTILOS_LINHA = [
    {'linestyle': '-', 'marker': 'o', 'label_style': 'Contínua'},
    {'linestyle': '--', 'marker': 's', 'label_style': 'Tracejada'},
    {'linestyle': '-.', 'marker': '^', 'label_style': 'Traço-ponto'},
    {'linestyle': ':', 'marker': 'D', 'label_style': 'Pontilhada'},
    {'linestyle': '-', 'marker': 'v', 'label_style': 'Contínua 2'},
    {'linestyle': '--', 'marker': 'p', 'label_style': 'Tracejada 2'},
    {'linestyle': '-.', 'marker': '*', 'label_style': 'Traço-ponto 2'},
    {'linestyle': ':', 'marker': 'h', 'label_style': 'Pontilhada 2'},
]


def formatar_numero_br(numero, casas=4):
    """Formata número substituindo ponto por vírgula"""
    return f"{numero:.{casas}f}".replace('.', ',')


print("="*70)
print("OR-SYSTEM - Comparação de Normalizações para TCC")
print("StandardScaler vs MinMaxScaler")
print("="*70)
print(f"Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")

# ============================================
# 1. Leitura dos Dados
# ============================================

dataset_bruto = pd.read_csv(
    "./or-system-dados-brutos.csv", decimal=",", thousands=".")

# ============================================
# GERAÇÃO DE GRÁFICOS COM DADOS BRUTOS
# ============================================
print("📊 Gerando gráficos com dados brutos...\n")

vventos_unicos = np.sort(dataset_bruto['vvento'].unique())

for vvento_val in vventos_unicos:
    dados_vvento = dataset_bruto[dataset_bruto['vvento'] == vvento_val]
    n_unicos = np.sort(dados_vvento['N'].unique())

    plt.figure(figsize=(6.3, 4.2))
    cores_bomba = plt.cm.tab10(np.linspace(0, 1, len(n_unicos)))

    for idx, n_val in enumerate(n_unicos):
        subset = dados_vvento[dados_vvento['N']
                              == n_val].sort_values('ang_virab')
        estilo = ESTILOS_LINHA[idx % len(ESTILOS_LINHA)]

        plt.plot(subset['ang_virab'], subset['pressao'],
                 marker=estilo['marker'],
                 linestyle=estilo['linestyle'],
                 linewidth=2, markersize=4,
                 color=cores_bomba[idx], alpha=0.7,
                 label=f'Rotação da bomba={formatar_numero_br(n_val)}')

    plt.xlabel('Ângulo do Virabrequim (rad)', fontsize=11)
    plt.ylabel('Pressão (bar)', fontsize=11)
    plt.title(
        f'Ângulo do Virabrequim vs Pressão - Velocidade do Vento = {formatar_numero_br(vvento_val)} m/s',
        fontsize=12, fontweight='bold')
    plt.legend(title='Rotação da Bomba (rpm)', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pressao_vs_angulo_vvento_dados_brutos_{vvento_val}.png",
                dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close()

print(f"✅ Gerados {len(vventos_unicos)} gráficos por velocidade de vento\n")

# DADOS TRATADOS
print("📂 Carregando dados...")
dataset = pd.read_csv("./or-system-database.csv", decimal=",", thousands=".")
x = dataset[["VPP", "ang_virab"]].values
y = dataset[["pressao"]].values
print(f"✅ Dados carregados: {len(x)} amostras\n")

vventos_unicos = np.sort(dataset['vvento'].unique())

for vvento_val in vventos_unicos:
    dados_vvento = dataset[dataset['vvento'] == vvento_val]
    n_unicos = np.sort(dados_vvento['N'].unique())

    plt.figure(figsize=(6.3, 4.2))
    cores_bomba = plt.cm.tab10(np.linspace(0, 1, len(n_unicos)))

    for idx, n_val in enumerate(n_unicos):
        subset = dados_vvento[dados_vvento['N']
                              == n_val].sort_values('ang_virab')
        estilo = ESTILOS_LINHA[idx % len(ESTILOS_LINHA)]

        plt.plot(subset['ang_virab'], subset['pressao'],
                 marker=estilo['marker'],
                 linestyle=estilo['linestyle'],
                 linewidth=2, markersize=4,
                 color=cores_bomba[idx], alpha=0.7,
                 label=f'Rotação da bomba={formatar_numero_br(n_val)}')

    plt.xlabel('Ângulo do Virabrequim (rad)', fontsize=11)
    plt.ylabel('Pressão (bar)', fontsize=11)
    plt.title(
        f'Ângulo do Virabrequim vs Pressão - Velocidade do Vento = {formatar_numero_br(vvento_val)} m/s',
        fontsize=12, fontweight='bold')
    plt.legend(title='Rotação da Bomba (rpm)', bbox_to_anchor=(
        1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pressao_vs_angulo_vvento_{vvento_val}.png",
                dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close()

print(
    f"Gerados {len(vventos_unicos)} gráficos, um para cada velocidade de vento.\n")

vpps_unicos = [0.3073, 0.5145, 0.5274, 0.7652]
plt.figure(figsize=(6.3, 4.2))

cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, vpp_val in enumerate(vpps_unicos):
    dados_vpp = dataset[np.isclose(dataset['VPP'], vpp_val, atol=0.0001)]
    dados_vpp = dados_vpp.sort_values('ang_virab')
    estilo = ESTILOS_LINHA[i % len(ESTILOS_LINHA)]

    plt.plot(dados_vpp['ang_virab'], dados_vpp['pressao'],
             marker=estilo['marker'],
             linestyle=estilo['linestyle'],
             linewidth=2, markersize=4,
             color=cores[i], alpha=0.7,
             label=f'VPP = {formatar_numero_br(vpp_val)}')

plt.xlabel('Ângulo do Virabrequim (rad)', fontsize=11)
plt.ylabel('Pressão (bar)', fontsize=11)
plt.title('Ângulo do Virabrequim vs Pressão para Diferentes VPP',
          fontsize=12, fontweight='bold')
plt.legend(title='Velocidade de ponta de pá', bbox_to_anchor=(
    1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("pressao_vs_angulo_vpps_especificos.png", dpi=600,
            bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Gráfico salvo: pressao_vs_angulo_vpps_especificos.png")

print("\n" + "="*60)
print("📊 ESTATÍSTICAS POR VPP")
print("="*60)


def alinhar_curvas(dataset, vvento_alvo):
    """Alinha (remove defasagem) das curvas para uma velocidade do vento específica."""
    df_vvento = dataset[dataset["vvento"] == vvento_alvo].copy()
    rotacoes = sorted(df_vvento["N"].unique())

    picos = {
        N: df_vvento[df_vvento["N"] == N]["pressao"].max()
        for N in rotacoes
    }
    ref_N = max(picos, key=picos.get)

    df_ref = df_vvento[df_vvento["N"] == ref_N]
    ang_ref = df_ref.loc[df_ref["pressao"].idxmax(), "ang_virab"]

    dfs_corrigidos = []

    for N in rotacoes:
        df_rot = df_vvento[df_vvento["N"] == N].copy()
        ang_pico = df_rot.loc[df_rot["pressao"].idxmax(), "ang_virab"]
        shift = ang_ref - ang_pico
        df_rot["ang_virab_corrigido"] = df_rot["ang_virab"] + shift
        dfs_corrigidos.append(df_rot)

    df_corrigido = pd.concat(dfs_corrigidos, ignore_index=True)
    return df_corrigido


velocidades_alvo = [3.5, 4.5, 5.5]

lista_corrigidos = []
for vv in velocidades_alvo:
    print(f"🔧 Aplicando alinhamento (shift) para vvento = {vv} m/s...")
    df_corr = alinhar_curvas(dataset, vv)
    lista_corrigidos.append(df_corr)

dataset_shifted = pd.concat(lista_corrigidos, ignore_index=True)

print("✅ Alinhamento concluído!")
print(f"➡ Tamanho do dataset original: {len(dataset)}")
print(f"➡ Tamanho do dataset corrigido: {len(dataset_shifted)}")
print("✓ Variável pronta: dataset_shifted\n")

vventos_unicos_shift = sorted(dataset_shifted["vvento"].unique())

for vvento_val in vventos_unicos_shift:
    dados_vvento = dataset_shifted[dataset_shifted['vvento'] == vvento_val]
    n_unicos = np.sort(dados_vvento['N'].unique())

    plt.figure(figsize=(6.3, 4.2))
    cores_bomba = plt.cm.tab10(np.linspace(0, 1, len(n_unicos)))

    for idx, n_val in enumerate(n_unicos):
        subset = dados_vvento[dados_vvento['N'] == n_val]
        subset = subset.sort_values("ang_virab_corrigido")
        estilo = ESTILOS_LINHA[idx % len(ESTILOS_LINHA)]

        plt.plot(subset['ang_virab_corrigido'], subset['pressao'],
                 marker=estilo['marker'],
                 linestyle=estilo['linestyle'],
                 linewidth=2, markersize=4,
                 color=cores_bomba[idx], alpha=0.7,
                 label=f'Rotação da bomba = {formatar_numero_br(n_val)}')

    plt.xlabel('Ângulo do Virabrequim Corrigido (rad)', fontsize=11)
    plt.ylabel('Pressão (bar)', fontsize=11)
    plt.title(f'Pressão vs Ângulo (Shift Aplicado) - vvento = {formatar_numero_br(vvento_val)} m/s',
              fontsize=12, fontweight='bold')
    plt.legend(title='Rotação da Bomba (rpm)', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"shift_pressao_vs_angulo_vvento_{vvento_val}.png"
    plt.savefig(fname, dpi=600, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close()

print(
    f"✅ Gerados {len(vventos_unicos_shift)} gráficos shiftados (por vvento).")

vpps_unicos = [0.3073, 0.5145, 0.5274, 0.7652]
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure(figsize=(6.3, 4.2))

for i, vpp_val in enumerate(vpps_unicos):
    dados_vpp = dataset_shifted[np.isclose(
        dataset_shifted['VPP'], vpp_val, atol=0.0001)]
    dados_vpp = dados_vpp.sort_values('ang_virab_corrigido')
    estilo = ESTILOS_LINHA[i % len(ESTILOS_LINHA)]

    plt.plot(
        dados_vpp['ang_virab_corrigido'],
        dados_vpp['pressao'],
        marker=estilo['marker'],
        linestyle=estilo['linestyle'],
        linewidth=2,
        markersize=4,
        color=cores[i],
        alpha=0.7,
        label=f'VPP = {formatar_numero_br(vpp_val)}'
    )

plt.xlabel('Ângulo do Virabrequim Corrigido (rad)', fontsize=11)
plt.ylabel('Pressão (bar)', fontsize=11)
plt.title('Pressão vs Ângulo para Diferentes VPP (Shift Aplicado)',
          fontsize=12, fontweight='bold')
plt.legend(title='Velocidade de ponta de pá',
           bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("shift_pressao_vs_angulo_vpps_especificos.png",
            dpi=600, bbox_inches='tight', pad_inches=0.05,
            facecolor='white')

print("✅ Gráfico salvo: shift_pressao_vs_angulo_vpps_especificos.png\n")

# ============================================
# 2. Definição das Configurações
# ============================================
print("🔧 Configurações a serem testadas:")
print("-" * 70)

mlp_configs = [
    {'name': 'MLP1', 'activation': 'tanh',
        'learning_rate': 0.01, 'verbose': False},
    {'name': 'MLP2', 'activation': 'logistic',
        'learning_rate': 0.01, 'verbose': False},
    {'name': 'MLP3', 'activation': 'relu',
        'learning_rate': 0.01, 'verbose': False},
    {'name': 'MLP4', 'activation': 'tanh',
        'learning_rate': 0.005, 'verbose': False},
    {'name': 'MLP5', 'activation': 'logistic',
        'learning_rate': 0.005, 'verbose': False},
    {'name': 'MLP6', 'activation': 'relu',
        'learning_rate': 0.005, 'verbose': False}
]

svm_configs = [
    {'name': 'SVM1', 'kernel': 'rbf', 'C': 1, 'epsilon': 0.1},
    {'name': 'SVM2', 'kernel': 'linear', 'C': 1, 'epsilon': 0.1},
    {'name': 'SVM3', 'kernel': 'poly',
        'degree': 3, 'C': 1, 'epsilon': 0.1},
    {'name': 'SVM4', 'kernel': 'rbf', 'C': 1, 'epsilon': 0.3},
    {'name': 'SVM5', 'kernel': 'linear', 'C': 1, 'epsilon': 0.3},
    {'name': 'SVM6', 'kernel': 'poly',
        'degree': 3, 'C': 1, 'epsilon': 0.3}
]

rl_configs = [
    {'name': 'RL1', 'penalty': 'l2', 'alpha': 0.001},
    {'name': 'RL2', 'penalty': 'l1', 'alpha': 0.0001},
    {'name': 'RL3', 'penalty': 'l1', 'alpha': 0.001},
    {'name': 'RL4', 'penalty': 'l2', 'alpha': 0.01},
    {'name': 'RL5', 'penalty': 'l1', 'alpha': 0.01},
    {'name': 'RL6', 'penalty': None, 'alpha': 0.0}
]

normalizadores = [
    {'nome': 'MinMax', 'scaler_class': MinMaxScaler},
    {'nome': 'Standard', 'scaler_class': StandardScaler}
]

print(f"  MLP: {len(mlp_configs)} configurações")
print(f"  SVM: {len(svm_configs)} configurações")
print(f"  RL:  {len(rl_configs)} configurações")
print(
    f"  Normalizadores: {len(normalizadores)} (StandardScaler, MinMaxScaler)")
print(f"\n  TOTAL: {(len(mlp_configs) + len(svm_configs) + len(rl_configs)) * len(normalizadores)} combinações")
print("-" * 70 + "\n")

# ============================================
# 3. Funções de Criação dos Modelos
# ============================================


def criar_mlp(activation, learning_rate, seed, verbose):
    return MLPRegressor(
        hidden_layer_sizes=(20),
        early_stopping=True,
        max_iter=1000,
        n_iter_no_change=100,
        tol=0.001,
        learning_rate_init=learning_rate,
        solver="sgd",
        activation=activation,
        learning_rate="constant",
        validation_fraction=0.2,
        verbose=verbose,
        shuffle=True,
        alpha=0.001,
        random_state=seed,
        batch_size=32
    )


def criar_svm(kernel, C, epsilon, degree=3):
    if kernel == 'poly':
        return SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree,
                   tol=0.001, max_iter=-1, verbose=0)
    else:
        return SVR(kernel=kernel, C=C, epsilon=epsilon,
                   tol=0.001, max_iter=-1, verbose=0)


def criar_reglinear(penalty, alpha):
    if penalty is None or alpha == 0.0:
        return LinearRegression()
    elif penalty == 'l2':
        return Ridge(alpha=alpha, tol=0.001)
    elif penalty == 'l1':
        return Lasso(alpha=alpha, tol=0.001)
    else:
        raise ValueError(f"Penalty '{penalty}' não reconhecido.")

# ============================================
# 4. Função de Avaliação COM PRINT
# ============================================


def avaliar_modelo(modelo, x_train, x_test, y_train, y_test,
                   scaler_y, nome_modelo, norm_nome):
    modelo.fit(x_train, y_train.ravel())

    pred_train_norm = modelo.predict(x_train)
    pred_test_norm = modelo.predict(x_test)

    pred_train = scaler_y.inverse_transform(pred_train_norm.reshape(-1, 1))
    pred_test = scaler_y.inverse_transform(pred_test_norm.reshape(-1, 1))

    y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    metricas = {
        'rmse_train': root_mean_squared_error(y_train_original, pred_train),
        'mae_train': mean_absolute_error(y_train_original, pred_train),
        'rmse_test': root_mean_squared_error(y_test_original, pred_test),
        'mae_test': mean_absolute_error(y_test_original, pred_test),
        'sucesso': True,
        'pred_test': pred_test.flatten(),
        'y_test': y_test_original.flatten()
    }
    return metricas


# ============================================
# 5. Execução dos Experimentos
# ============================================
n_rodadas = 50
resultados_completos = []
predicoes_guardadas = {}  # Para armazenar predições dos melhores modelos

print("🔬 Iniciando experimentos...\n")

for rodada in range(n_rodadas):
    print(f"\n{'='*70}")
    print(f"RODADA {rodada + 1}/{n_rodadas}")
    print(f"{'='*70}")

    seed = 20000+rodada
    np.random.seed(seed)
    indices = np.random.permutation(len(x))
    x_emb = x[indices]
    y_emb = y[indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x_emb, y_emb, test_size=0.2, random_state=seed
    )

    resultado_rodada = {'rodada': rodada + 1}

    for norm_config in normalizadores:
        scaler_x = norm_config['scaler_class']()
        scaler_y = norm_config['scaler_class']()

        x_train_norm = scaler_x.fit_transform(x_train)
        x_test_norm = scaler_x.transform(x_test)
        y_train_norm = scaler_y.fit_transform(y_train)
        y_test_norm = scaler_y.transform(y_test)

        norm_nome = norm_config['nome']

        # Testar MLPs
        for config in mlp_configs:
            mlp = criar_mlp(
                config['activation'], config['learning_rate'], rodada, config['verbose'])
            metricas = avaliar_modelo(mlp, x_train_norm, x_test_norm,
                                      y_train_norm, y_test_norm, scaler_y,
                                      config['name'], norm_nome)

            # MODIFICAÇÃO: Guardar predições apenas para Standard
            if rodada == 0 and metricas['sucesso'] and norm_nome == 'Standard':
                chave = f"{config['name']}_{norm_nome}"
                predicoes_guardadas[chave] = {
                    'pred': metricas['pred_test'],
                    'real': metricas['y_test'],
                    'x_test': x_test
                }

            for nome_metrica, valor in metricas.items():
                if nome_metrica not in ['sucesso', 'pred_test', 'y_test']:
                    metrica_tipo = nome_metrica.split('_')[0].upper()
                    if 'train' in nome_metrica:
                        conjunto = 'Treino'
                    elif 'test' in nome_metrica:
                        conjunto = 'Teste'
                    else:
                        continue
                    col_name = f"{metrica_tipo}_{config['name']}_{norm_nome}_{conjunto}"
                    resultado_rodada[col_name] = valor

        # Testar SVMs
        for config in svm_configs:
            svm = criar_svm(config['kernel'], config['C'], config['epsilon'],
                            config.get('degree', 2))
            metricas = avaliar_modelo(svm, x_train_norm, x_test_norm,
                                      y_train_norm, y_test_norm, scaler_y,
                                      config['name'], norm_nome)

            # MODIFICAÇÃO: Guardar predições apenas para Standard
            if rodada == 0 and metricas['sucesso'] and norm_nome == 'Standard':
                chave = f"{config['name']}_{norm_nome}"
                predicoes_guardadas[chave] = {
                    'pred': metricas['pred_test'],
                    'real': metricas['y_test'],
                    'x_test': x_test
                }

            for nome_metrica, valor in metricas.items():
                if nome_metrica not in ['sucesso', 'pred_test', 'y_test']:
                    metrica_tipo = nome_metrica.split('_')[0].upper()
                    if 'train' in nome_metrica:
                        conjunto = 'Treino'
                    elif 'test' in nome_metrica:
                        conjunto = 'Teste'
                    else:
                        continue
                    col_name = f"{metrica_tipo}_{config['name']}_{norm_nome}_{conjunto}"
                    resultado_rodada[col_name] = valor

        # Testar RLs
        for config in rl_configs:
            rl = criar_reglinear(config['penalty'], config['alpha'])
            metricas = avaliar_modelo(rl, x_train_norm, x_test_norm,
                                      y_train_norm, y_test_norm, scaler_y,
                                      config['name'], norm_nome)

            # MODIFICAÇÃO: Guardar predições apenas para Standard
            if rodada == 0 and metricas['sucesso'] and norm_nome == 'Standard':
                chave = f"{config['name']}_{norm_nome}"
                predicoes_guardadas[chave] = {
                    'pred': metricas['pred_test'],
                    'real': metricas['y_test'],
                    'x_test': x_test
                }

            for nome_metrica, valor in metricas.items():
                if nome_metrica not in ['sucesso', 'pred_test', 'y_test']:
                    metrica_tipo = nome_metrica.split('_')[0].upper()
                    if 'train' in nome_metrica:
                        conjunto = 'Treino'
                    elif 'test' in nome_metrica:
                        conjunto = 'Teste'
                    else:
                        continue
                    col_name = f"{metrica_tipo}_{config['name']}_{norm_nome}_{conjunto}"
                    resultado_rodada[col_name] = valor

    resultados_completos.append(resultado_rodada)

print(f"\n\n✅ Todos os experimentos concluídos!\n")

# ============================================
# 6. Processamento dos Resultados
# ============================================
df_resultados = pd.DataFrame(resultados_completos)

# ============================================
# 7. Análise Estatística
# ============================================
print("\n" + "="*70)
print("📊 ANÁLISE COMPARATIVA DE NORMALIZAÇÕES")
print("="*70 + "\n")

todos_modelos = []
for config in mlp_configs:
    todos_modelos.append(config['name'])
for config in svm_configs:
    todos_modelos.append(config['name'])
for config in rl_configs:
    todos_modelos.append(config['name'])

tabelas_por_norm = {}

for norm_config in normalizadores:
    norm_nome = norm_config['nome']
    tabela_resumo = []

    for modelo in todos_modelos:
        linha = {'Modelo': modelo.replace('_', ' '), 'Normalizador': norm_nome}
        for metrica in ['RMSE', 'MAE']:
            col_teste = f"{metrica}_{modelo}_{norm_nome}_Teste"
            col_treino = f"{metrica}_{modelo}_{norm_nome}_Treino"
            if col_teste in df_resultados.columns:
                media_teste = df_resultados[col_teste].mean()
                std_teste = df_resultados[col_teste].std()
                linha[f"{metrica}_Teste_Média"] = media_teste
                linha[f"{metrica}_Teste_DP"] = std_teste
                if col_treino in df_resultados.columns:
                    media_treino = df_resultados[col_treino].mean()
                    std_treino = df_resultados[col_treino].std()
                    linha[f"{metrica}_Treino_Média"] = media_treino
                    linha[f"{metrica}_Treino_DP"] = std_treino
        tabela_resumo.append(linha)

    tabelas_por_norm[norm_nome] = pd.DataFrame(tabela_resumo)

df_resumo_completo = pd.concat(tabelas_por_norm.values(), ignore_index=True)
df_resumo_completo.to_csv("tabela_resumo_normalizacoes.csv",
                          index=False, sep=';', decimal=',')
print("💾 Arquivo salvo: tabela_resumo_normalizacoes.csv")

# ============================================
# 8. GRÁFICOS DE PREDIÇÃO vs REAL (APENAS STANDARD)
# ============================================
print("\n" + "="*70)
print("📊 GERANDO GRÁFICOS DE PREDIÇÃO vs VALORES REAIS (StandardScaler)")
print("="*70 + "\n")

# Identificar o melhor modelo de cada tipo (MLP, SVM, RL) - APENAS STANDARD
tipos_modelos = {
    'MLP': [],
    'SVM': [],
    'RL': []
}

# Classificar modelos por tipo - APENAS STANDARD
df_norm_std = tabelas_por_norm['Standard']

for _, row in df_norm_std.iterrows():
    modelo = row['Modelo']
    if 'MLP' in modelo:
        tipos_modelos['MLP'].append((modelo, 'Standard', row['RMSE_Teste_Média']))
    elif 'SVM' in modelo:
        tipos_modelos['SVM'].append((modelo, 'Standard', row['RMSE_Teste_Média']))
    elif 'RL' in modelo:
        tipos_modelos['RL'].append((modelo, 'Standard', row['RMSE_Teste_Média']))

# Encontrar o melhor de cada tipo
melhores_por_tipo = {}
for tipo, lista in tipos_modelos.items():
    if lista:
        melhor = min(lista, key=lambda x: x[2])  # Pega o menor RMSE
        melhores_por_tipo[tipo] = {
            'modelo': melhor[0],
            'norm': melhor[1],
            'rmse': melhor[2]
        }

print("🏆 Melhores modelos selecionados (StandardScaler):")
for tipo, info in melhores_por_tipo.items():
    print(f"  {tipo}: {info['modelo']} - RMSE: {formatar_numero_br(info['rmse'])}")

# Criar gráficos para os melhores modelos de cada tipo
print("\n📈 Gerando gráficos individuais...\n")

for tipo, info in melhores_por_tipo.items():
    modelo_nome = info['modelo']
    norm_nome = info['norm']
    modelo_key = modelo_nome.replace(' ', '_')
    chave = f"{modelo_key}_{norm_nome}"

    if chave in predicoes_guardadas:
        dados = predicoes_guardadas[chave]

        # Criar figura com 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Predições - Melhor {tipo}: {modelo_nome} [StandardScaler]',
                     fontsize=14, fontweight='bold')

        # Subplot 1: Scatter plot (Predito vs Real)
        ax1.scatter(dados['real'], dados['pred'], alpha=0.6, s=20)

        # Linha ideal (y=x)
        min_val = min(dados['real'].min(), dados['pred'].min())
        max_val = max(dados['real'].max(), dados['pred'].max())
        ax1.plot([min_val, max_val], [min_val, max_val],
                 'r--', linewidth=2, label='Predição Perfeita')

        ax1.set_xlabel('Pressão Real (bar)', fontsize=11)
        ax1.set_ylabel('Pressão Predita (bar)', fontsize=11)
        ax1.set_title('Correlação: Predito vs Real', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')

        # Subplot 2: Série temporal
        indices_ordenados = np.argsort(dados['real'])
        ax2.plot(dados['real'][indices_ordenados],
                 label='Valor Real', linewidth=2, marker='o',
                 markersize=3, alpha=0.7)
        ax2.plot(dados['pred'][indices_ordenados],
                 label='Valor Predito', linewidth=2, marker='s',
                 markersize=3, alpha=0.7)

        ax2.set_xlabel('Índice (ordenado por valor real)', fontsize=11)
        ax2.set_ylabel('Pressão (bar)', fontsize=11)
        ax2.set_title('Comparação Sequencial', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Salvar
        nome_arquivo = f"predicao_melhor_{tipo}_{modelo_key}_Standard.png"
        plt.savefig(nome_arquivo, dpi=600, bbox_inches='tight',
                    pad_inches=0.05, facecolor='white')
        plt.close()
        print(f"✅ Salvo: {nome_arquivo}")


# ============================================
# 9. Visualizações de Rankings
# ============================================

# FIGURA 1: Ranking Geral - CORRIGIDO
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 3.8))

ranking_std = tabelas_por_norm['Standard'].sort_values('RMSE_Teste_Média')
colors1 = plt.cm.viridis(np.linspace(0.2, 0.9, len(ranking_std)))
ax1.barh(range(len(ranking_std)),
         ranking_std['RMSE_Teste_Média'], color=colors1)
ax1.set_yticks(range(len(ranking_std)))
ax1.set_yticklabels(ranking_std['Modelo'], fontsize=9)
ax1.set_xlabel('RMSE Médio', fontsize=11)
ax1.set_title('StandardScaler', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')
# MODIFICAÇÃO: Posicionar os números fora das barras
max_val_std = ranking_std['RMSE_Teste_Média'].max()
for i, (idx, row) in enumerate(ranking_std.iterrows()):
    ax1.text(row['RMSE_Teste_Média'] + max_val_std * 0.02, i, 
             f"{formatar_numero_br(row['RMSE_Teste_Média'])}",
             va='center', fontsize=8, ha='left')

ranking_mm = tabelas_por_norm['MinMax'].sort_values('RMSE_Teste_Média')
colors2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(ranking_mm)))
ax2.barh(range(len(ranking_mm)), ranking_mm['RMSE_Teste_Média'], color=colors2)
ax2.set_yticks(range(len(ranking_mm)))
ax2.set_yticklabels(ranking_mm['Modelo'], fontsize=9)
ax2.set_xlabel('RMSE Médio', fontsize=11)
ax2.set_title('MinMaxScaler', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
# MODIFICAÇÃO: Posicionar os números fora das barras
max_val_mm = ranking_mm['RMSE_Teste_Média'].max()
for i, (idx, row) in enumerate(ranking_mm.iterrows()):
    ax2.text(row['RMSE_Teste_Média'] + max_val_mm * 0.02, i, 
             f"{formatar_numero_br(row['RMSE_Teste_Média'])}",
             va='center', fontsize=8, ha='left')

# MODIFICAÇÃO: Ajustar limites do eixo x para acomodar os textos
ax1.set_xlim(0, max_val_std * 1.3)
ax2.set_xlim(0, max_val_mm * 1.3)

plt.tight_layout()
plt.savefig("figura1_ranking_normalizacoes.png", dpi=600,
            bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("\n✅ Salvo: figura1_ranking_normalizacoes.png")

# ============================================
# VISUALIZAÇÕES ADICIONAIS PARA O TCC
# ============================================

print("\n" + "="*70)
print("📊 GERANDO VISUALIZAÇÕES ADICIONAIS PARA ANÁLISE")
print("="*70 + "\n")

# FIGURA 3: ANÁLISE DE ESTABILIDADE (CV) - CORRIGIDA (APENAS STANDARD)
print("📈 Gerando análise de estabilidade...")

fig7, ax = plt.subplots(1, 1, figsize=(6.3, 3.8))

norm_nome = 'Standard'

cvs = []
labels = []

for modelo in todos_modelos:
    row = tabelas_por_norm[norm_nome][
        tabelas_por_norm[norm_nome]['Modelo'] == modelo.replace('_', ' ')
    ].iloc[0]

    cv = (row['RMSE_Teste_DP'] / row['RMSE_Teste_Média']) * 100
    cvs.append(cv)
    labels.append(modelo.replace('_', ' '))

sorted_indices = np.argsort(cvs)
cvs_sorted = [cvs[i] for i in sorted_indices]
labels_sorted = [labels[i] for i in sorted_indices]

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cvs_sorted)))

bars = ax.barh(range(len(cvs_sorted)), cvs_sorted, color=colors, alpha=0.8)
ax.set_yticks(range(len(labels_sorted)))
ax.set_yticklabels(labels_sorted, fontsize=9)
ax.set_xlabel('Coeficiente de Variação (%)', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# MODIFICAÇÃO: Posicionar números fora das barras com 2 casas decimais
max_cv = max(cvs_sorted) if cvs_sorted else 1.0
# CORREÇÃO: Verificar se max_cv é válido
if np.isnan(max_cv) or np.isinf(max_cv) or max_cv == 0:
    max_cv = 1.0

for i, cv in enumerate(cvs_sorted):
    # Formatar com 2 casas decimais
    cv_formatado = formatar_numero_br(cv, casas=2)
    ax.text(cv + max_cv * 0.02, i, f'{cv_formatado}%',
            va='center', fontsize=8, ha='left')

# MODIFICAÇÃO: Ajustar limite do eixo x com verificação
ax.set_xlim(0, max_cv * 1.15)

plt.tight_layout()
plt.savefig("figura3_estabilidade_modelos.png", dpi=600,
            bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura3_estabilidade_modelos.png")

# FIGURA 4: COMPARAÇÃO DE HIPERPARÂMETROS - MLP - CORRIGIDA (APENAS STANDARD)
print("📈 Gerando comparação de hiperparâmetros MLP...")

# MODIFICAÇÃO: Aumentar altura para acomodar legenda
fig9, ax = plt.subplots(1, 1, figsize=(6.3, 4.5))

norm_nome = 'Standard'

# Mapear configurações MLP
mlp_map = {
    'MLP1': {'activation': 'tanh', 'lr': 0.01},
    'MLP2': {'activation': 'logistic', 'lr': 0.01},
    'MLP3': {'activation': 'relu', 'lr': 0.01},
    'MLP4': {'activation': 'tanh', 'lr': 0.005},
    'MLP5': {'activation': 'logistic', 'lr': 0.005},
    'MLP6': {'activation': 'relu', 'lr': 0.005}
}

ativacoes = ['tanh', 'logistic', 'relu']
lrs = [0.005, 0.01]

dados_plot = {ativ: [] for ativ in ativacoes}

for ativ in ativacoes:
    for lr in lrs:
        # Buscar pelos nomes MLP1-MLP6
        modelo_encontrado = None
        for mlp_nome, mlp_config in mlp_map.items():
            if mlp_config['activation'] == ativ and mlp_config['lr'] == lr:
                modelo_encontrado = mlp_nome
                break
        
        if modelo_encontrado:
            row = tabelas_por_norm[norm_nome][
                tabelas_por_norm[norm_nome]['Modelo'] == modelo_encontrado
            ]
            if not row.empty:
                dados_plot[ativ].append(row.iloc[0]['RMSE_Teste_Média'])

if all(len(dados_plot[a]) == len(lrs) for a in ativacoes):
    x = np.arange(len(lrs))
    width = 0.25

    cores_ativ = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # MODIFICAÇÃO: Adicionar hachuras para diferenciação em P&B
    hachuras = ['///', '\\\\\\', '|||']
    
    # MODIFICAÇÃO: Calcular valor máximo para ajuste
    max_val = max([max(dados_plot[a]) for a in ativacoes if dados_plot[a]])
    
    for i, ativ in enumerate(ativacoes):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, dados_plot[ativ], width,
                     label=ativ.capitalize(), alpha=0.8, color=cores_ativ[i],
                     hatch=hachuras[i], edgecolor='black', linewidth=0.5)
        
        # MODIFICAÇÃO: Posicionar números acima das barras
        for j, (pos, valor) in enumerate(zip(x + offset, dados_plot[ativ])):
            ax.text(pos, valor + max_val * 0.02, formatar_numero_br(valor), 
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Taxa de Aprendizado', fontsize=11)
    ax.set_ylabel('RMSE Teste Médio', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['0,005', '0,01'])
    
    # MODIFICAÇÃO: Posicionar legenda no topo
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    # MODIFICAÇÃO: Ajustar limite do eixo y
    ax.set_ylim(0, max_val * 1.15)
else:
    ax.text(0.5, 0.5, 'Dados insuficientes\npara gerar gráfico',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig("figura4_hiperparametros_mlp.png", dpi=600,
            bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura4_hiperparametros_mlp.png")

# FIGURA 5: COMPARAÇÃO DE KERNELS - SVM - CORRIGIDA (APENAS STANDARD)
print("📈 Gerando comparação de kernels SVM...")

# MODIFICAÇÃO: Aumentar altura para acomodar legenda
fig10, ax = plt.subplots(1, 1, figsize=(6.3, 4.5))

norm_nome = 'Standard'

# Mapear configurações SVM
svm_map = {
    'SVM1': {'kernel': 'rbf', 'epsilon': 0.1},
    'SVM2': {'kernel': 'linear', 'epsilon': 0.1},
    'SVM3': {'kernel': 'poly', 'epsilon': 0.1},
    'SVM4': {'kernel': 'rbf', 'epsilon': 0.3},
    'SVM5': {'kernel': 'linear', 'epsilon': 0.3},
    'SVM6': {'kernel': 'poly', 'epsilon': 0.3}
}

kernels = ['rbf', 'linear', 'poly']
epsilons = [0.1, 0.3]

dados_plot = {kernel: [] for kernel in kernels}

for kernel in kernels:
    for epsilon in epsilons:
        # Buscar pelos nomes SVM1-SVM6
        modelo_encontrado = None
        for svm_nome, svm_config in svm_map.items():
            if svm_config['kernel'] == kernel and svm_config['epsilon'] == epsilon:
                modelo_encontrado = svm_nome
                break
        
        if modelo_encontrado:
            row = tabelas_por_norm[norm_nome][
                tabelas_por_norm[norm_nome]['Modelo'] == modelo_encontrado
            ]
            if not row.empty:
                dados_plot[kernel].append(row.iloc[0]['RMSE_Teste_Média'])

if all(len(dados_plot[k]) == len(epsilons) for k in kernels):
    x = np.arange(len(epsilons))
    width = 0.25

    cores_kernel = ['#d62728', '#9467bd', '#8c564b']
    
    # MODIFICAÇÃO: Adicionar hachuras para diferenciação em P&B
    hachuras = ['///', '\\\\\\', '|||']
    
    # MODIFICAÇÃO: Calcular valor máximo para ajuste
    max_val = max([max(dados_plot[k]) for k in kernels if dados_plot[k]])
    
    for i, kernel in enumerate(kernels):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, dados_plot[kernel], width,
                     label=kernel.upper(), alpha=0.8, color=cores_kernel[i],
                     hatch=hachuras[i], edgecolor='black', linewidth=0.5)
        
        # MODIFICAÇÃO: Posicionar números acima das barras
        for j, (pos, valor) in enumerate(zip(x + offset, dados_plot[kernel])):
            ax.text(pos, valor + max_val * 0.02, formatar_numero_br(valor), 
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Epsilon', fontsize=11)
    ax.set_ylabel('RMSE Teste Médio', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['ε=0,1', 'ε=0,3'])
    
    # MODIFICAÇÃO: Posicionar legenda no topo
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    # MODIFICAÇÃO: Ajustar limite do eixo y
    ax.set_ylim(0, max_val * 1.15)
else:
    ax.text(0.5, 0.5, 'Dados insuficientes\npara gerar gráfico',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig("figura5_hiperparametros_svm.png", dpi=600,
            bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura5_hiperparametros_svm.png")

# FIGURA 6: COMPARAÇÃO REGULARIZAÇÃO - RL - CORRIGIDA (APENAS STANDARD)
print("📈 Gerando comparação de regularização RL...")

fig11, ax = plt.subplots(1, 1, figsize=(6.3, 3.8))

norm_nome = 'Standard'

modelos_rl = ['RL1', 'RL2', 'RL3', 'RL4', 'RL5', 'RL6']

# Mapear modelos RL
rl_descricoes = {
    'RL1': 'Ridge\nα=0,001',
    'RL2': 'Lasso\nα=0,0001',
    'RL3': 'Lasso\nα=0,001',
    'RL4': 'Ridge\nα=0,01',
    'RL5': 'Lasso\nα=0,01',
    'RL6': 'MQO'
}

valores = []
labels_curtos = []

for modelo in modelos_rl:
    row = tabelas_por_norm[norm_nome][
        tabelas_por_norm[norm_nome]['Modelo'] == modelo
    ]
    if not row.empty:
        valores.append(row.iloc[0]['RMSE_Teste_Média'])
        labels_curtos.append(rl_descricoes.get(modelo, modelo))

# MODIFICAÇÃO: Verificar se há valores antes de plotar
if valores:
    # Cores: MQO em cinza, Ridge em azul, Lasso em vermelho
    colors = []
    for modelo in modelos_rl[:len(valores)]:
        if modelo == 'RL6':
            colors.append('gray')
        elif modelo in ['RL1', 'RL4']:
            colors.append('blue')
        else:
            colors.append('red')
    
    bars = ax.bar(range(len(valores)), valores, color=colors, alpha=0.7)

    ax.set_xticks(range(len(labels_curtos)))
    ax.set_xticklabels(labels_curtos, fontsize=9)
    ax.set_ylabel('RMSE Teste Médio', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # MODIFICAÇÃO: Posicionar números acima das barras
    max_val = max(valores)
    for i, v in enumerate(valores):
        ax.text(i, v + max_val * 0.02, formatar_numero_br(v), 
                ha='center', va='bottom', fontsize=9)
    
    # MODIFICAÇÃO: Ajustar limite do eixo y
    ax.set_ylim(0, max_val * 1.15)
else:
    ax.text(0.5, 0.5, 'Dados insuficientes\npara gerar gráfico',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig("figura6_regularizacao_rl.png", dpi=600,
            bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura6_regularizacao_rl.png")

# RELATÓRIO FINAL
print("\n" + "="*70)
print("✅ ANÁLISE DE NORMALIZAÇÃO FINALIZADA!")
print("="*70)
print(f"\nTempo de execução: {datetime.now().strftime('%H:%M:%S')}")
print("\n📁 Arquivos gerados:")
print("   1. tabela_resumo_normalizacoes.csv")
print("   2. figura1_ranking_normalizacoes.png (CORRIGIDA)")
print("   3. figura2_impacto_normalizacao.png")
print("   4. figura3_estabilidade_modelos.png")
print("   5. figura4_hiperparametros_mlp.png")
print("   6. figura5_hiperparametros_svm.png")
print("   7. figura6_regularizacao_rl.png")
print(f"   8-10. Gráficos de predição dos 3 melhores modelos (APENAS StandardScaler)")
print(f"   11+. Gráficos de pressão vs ângulo (vvento e VPP)")
print("="*70)
print(f"\n✨ Execução finalizada: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*70)