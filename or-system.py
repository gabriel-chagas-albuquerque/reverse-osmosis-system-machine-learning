# ============================================
# OR-SYSTEM - Teste Focado com 2 Normalizações
# StandardScaler vs MinMaxScaler
# COM PRINTS DETALHADOS E TABELAS VISUAIS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração visual
plt.rcParams.update({
    'figure.dpi': 600,          # alta definição na tela
    'savefig.dpi': 600,         # alta definição ao salvar
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    'font.family': 'serif',
    'font.size': 10,            # tamanho de texto padrão para TCC
})

print("="*70)
print("OR-SYSTEM - Comparação de Normalizações para TCC")
print("StandardScaler vs MinMaxScaler")
print("="*70)
print(f"Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")

# ============================================
# 1. Leitura dos Dados
# ============================================

dataset_bruto = pd.read_csv("./or-system-dados-brutos.csv", decimal=",", thousands=".")

# ============================================
# GERAÇÃO DE GRÁFICOS COM DADOS BRUTOS
# ============================================
print("📊 Gerando gráficos com dados brutos...\n")

# Obtém as velocidades de vento únicas
vventos_unicos = np.sort(dataset_bruto['vvento'].unique())

# Cria um gráfico para cada velocidade de vento
for vvento_val in vventos_unicos:
    dados_vvento = dataset_bruto[dataset_bruto['vvento'] == vvento_val]
    n_unicos = np.sort(dados_vvento['N'].unique())

    plt.figure(figsize=(6.3, 4.2))
    
    # Cores distintas para cada rotação da bomba
    cores_bomba = plt.cm.tab10(np.linspace(0, 1, len(n_unicos)))
    
    for idx, n_val in enumerate(n_unicos):
        subset = dados_vvento[dados_vvento['N'] == n_val].sort_values('ang_virab')
        plt.plot(subset['ang_virab'], subset['pressao'],
                 marker='o', linestyle='-', linewidth=2, markersize=2,
                 color=cores_bomba[idx], alpha=0.7,
                 label=f'Rotação da bomba={n_val:.2f}')

    plt.xlabel('Ângulo do Virabrequim (rad)', fontsize=11)
    plt.ylabel('Pressão (bar)', fontsize=11)
    plt.title(
        f'Ângulo do Virabrequim vs Pressão - Velocidade do Vento = {vvento_val} m/s',
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

# Obtém as velocidades de vento únicas
vventos_unicos = np.sort(dataset['vvento'].unique())

# Cria um gráfico para cada velocidade de vento
for vvento_val in vventos_unicos:
    dados_vvento = dataset[dataset['vvento'] == vvento_val]
    n_unicos = np.sort(dados_vvento['N'].unique())

    plt.figure(figsize=(6.3, 4.2))
    for n_val in n_unicos:
        subset = dados_vvento[dados_vvento['N'] == n_val]
        plt.plot(subset['ang_virab'], subset['pressao'],
                 marker='o', linestyle='-', label=f'Rotação da bomba={n_val}')

    plt.xlabel('Ângulo do Virabrequim (rad)')
    plt.ylabel('Pressão (bar)')
    plt.title(
        f'Ângulo do Virabrequim vs Pressão - Velocidade do Vento = {vvento_val} m/s')
    plt.legend(title='Rotação da Bomba (rpm)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pressao_vs_angulo_vvento_{vvento_val}.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close()

print(
    f"Gerados {len(vventos_unicos)} gráficos, um para cada velocidade de vento.\n")

vpps_unicos = [0.3073, 0.5145, 0.5274, 0.7652]
plt.figure(figsize=(6.3, 4.2))

# Cores para cada VPP
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plotar dados para cada VPP
for i, vpp_val in enumerate(vpps_unicos):
    # Filtrar dados para o VPP específico
    dados_vpp = dataset[np.isclose(dataset['VPP'], vpp_val, atol=0.0001)]
    
    # Ordenar por ângulo do virabrequim para linha contínua
    dados_vpp = dados_vpp.sort_values('ang_virab')
    
    # Plotar
    plt.plot(dados_vpp['ang_virab'], dados_vpp['pressao'],
             marker='o', linestyle='-', linewidth=2, markersize=4,
             color=cores[i], alpha=0.7, label=f'VPP = {vpp_val:.4f}')

plt.xlabel('Ângulo do Virabrequim (rad)', fontsize=11)
plt.ylabel('Pressão (bar)', fontsize=11)
plt.title('Ângulo do Virabrequim vs Pressão para Diferentes VPP', fontsize=12, fontweight='bold', )
plt.legend(title='Velocidade de ponta de pá', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvar
plt.savefig("pressao_vs_angulo_vpps_especificos.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Gráfico salvo: pressao_vs_angulo_vpps_especificos.png")

# Estatísticas por VPP
print("\n" + "="*60)
print("📊 ESTATÍSTICAS POR VPP")
print("="*60)

def alinhar_curvas(dataset, vvento_alvo):
    """
    Alinha (remove defasagem) das curvas para uma velocidade do vento específica.
    Retorna um dataset corrigido apenas para aquela velocidade.
    """

    df_vvento = dataset[dataset["vvento"] == vvento_alvo].copy()

    # Descobrir todas as rotações presentes
    rotações = sorted(df_vvento["N"].unique())

    # Encontrar curva de referência = maior pico de pressão (amplitude maior)
    picos = {
        N: df_vvento[df_vvento["N"] == N]["pressao"].max()
        for N in rotações
    }
    ref_N = max(picos, key=picos.get)

    # Descobrir ÂNGULO do pico da referência
    df_ref = df_vvento[df_vvento["N"] == ref_N]
    ang_ref = df_ref.loc[df_ref["pressao"].idxmax(), "ang_virab"]

    # Nova lista de dataframes corrigidos
    dfs_corrigidos = []

    for N in rotações:
        df_rot = df_vvento[df_vvento["N"] == N].copy()

        # ângulo do pico dessa rotação
        ang_pico = df_rot.loc[df_rot["pressao"].idxmax(), "ang_virab"]

        # shift necessário
        shift = ang_ref - ang_pico

        # aplicar shift
        df_rot["ang_virab_corrigido"] = df_rot["ang_virab"] + shift

        dfs_corrigidos.append(df_rot)

    # retornar conjunto inteiro corrigido
    df_corrigido = pd.concat(dfs_corrigidos, ignore_index=True)
    return df_corrigido


# ------------------------------------------------------------
# CRIAR DATASET GLOBAL CORRIGIDO PARA AS 3 VELOCIDADES
# ------------------------------------------------------------

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
print("✔ Variável pronta: dataset_shifted\n")

vventos_unicos_shift = sorted(dataset_shifted["vvento"].unique())

for vvento_val in vventos_unicos_shift:
    dados_vvento = dataset_shifted[dataset_shifted['vvento'] == vvento_val]
    n_unicos = np.sort(dados_vvento['N'].unique())

    plt.figure(figsize=(6.3, 4.2))

    for n_val in n_unicos:
        subset = dados_vvento[dados_vvento['N'] == n_val]

        # Agora usando ang_virab_corrigido
        subset = subset.sort_values("ang_virab_corrigido")

        plt.plot(subset['ang_virab_corrigido'], subset['pressao'],
                 marker='o', linestyle='-', label=f'Rotação da bomba = {n_val}')

    plt.xlabel('Ângulo do Virabrequim Corrigido (rad)')
    plt.ylabel('Pressão (bar)')
    plt.title(f'Pressão vs Ângulo (Shift Aplicado) - vvento = {vvento_val} m/s')
    plt.legend(title='Rotação da Bomba (rpm)', bbox_to_anchor=(1.05, 1),
               loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"shift_pressao_vs_angulo_vvento_{vvento_val}.png"
    plt.savefig(fname, dpi=600, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close()

print(f"✅ Gerados {len(vventos_unicos_shift)} gráficos shiftados (por vvento).")


# ============================================================
#  GRÁFICO PARA VPPs ESPECÍFICOS (USANDO SHIFT)
# ============================================================

vpps_unicos = [0.3073, 0.5145, 0.5274, 0.7652]
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure(figsize=(6.3, 4.2))

for i, vpp_val in enumerate(vpps_unicos):

    dados_vpp = dataset_shifted[np.isclose(dataset_shifted['VPP'], vpp_val, atol=0.0001)]

    dados_vpp = dados_vpp.sort_values('ang_virab_corrigido')

    plt.plot(
        dados_vpp['ang_virab_corrigido'],
        dados_vpp['pressao'],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=4,
        color=cores[i],
        alpha=0.7,
        label=f'VPP = {vpp_val:.4f}'
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
    {'name': 'MLP_tanh_lr_0.01', 'activation': 'tanh',
        'learning_rate': 0.01, 'verbose': False},
    {'name': 'MLP_sigmoid_lr_0.01', 'activation': 'logistic',
        'learning_rate': 0.01, 'verbose': False},
    {'name': 'MLP_relu_lr_0.01', 'activation': 'relu',
        'learning_rate': 0.01, 'verbose': False},
    {'name': 'MLP_tanh_lr_0.005', 'activation': 'tanh',
        'learning_rate': 0.005, 'verbose': False},
    {'name': 'MLP_sigmoid_lr_0.005', 'activation': 'logistic',
        'learning_rate': 0.005, 'verbose': False},
    {'name': 'MLP_relu_lr_0.005', 'activation': 'relu',
        'learning_rate': 0.005, 'verbose': False}
]

svm_configs = [
    {'name': 'SVM_rbf_epsilon_0.1', 'kernel': 'rbf', 'C': 1, 'epsilon': 0.1},
    {'name': 'SVM_linear_epsilon_0.1', 'kernel': 'linear', 'C': 1, 'epsilon': 0.1},
    {'name': 'SVM_poly_epsilon_0.1', 'kernel': 'poly',
        'degree': 3, 'C': 1, 'epsilon': 0.1},
    {'name': 'SVM_rbf_epsilon_0.3', 'kernel': 'rbf', 'C': 1, 'epsilon': 0.3},
    {'name': 'SVM_linear_epsilon_0.3', 'kernel': 'linear', 'C': 1, 'epsilon': 0.3},
    {'name': 'SVM_poly_epsilon_0.3', 'kernel': 'poly',
        'degree': 3, 'C': 1, 'epsilon': 0.3}
]

rl_configs = [
    {'name': 'RL_MQO', 'penalty': None, 'alpha': 0.0},
    {'name': 'RL_Ridge_alpha_0.1', 'penalty': 'l2', 'alpha': 0.1},
    {'name': 'RL_Lasso_alpha_0.1', 'penalty': 'l1', 'alpha': 0.1},
    {'name': 'RL_Lasso_alpha_0.5', 'penalty': 'l1', 'alpha': 0.5},
    {'name': 'RL_Ridge_alpha_10', 'penalty': 'l2', 'alpha': 10},
    {'name': 'RL_Lasso_alpha_1', 'penalty': 'l1', 'alpha': 1}
]

normalizadores = [
    {'nome': 'Standard', 'scaler_class': StandardScaler},
    {'nome': 'MinMax', 'scaler_class': MinMaxScaler}
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
        return LinearRegression(tol=0.001)
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
    try:
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
            'sucesso': True
        }

        # PRINT DETALHADO
        print(f"\n  📊 {nome_modelo} [{norm_nome}]")
        print(
            f"      TREINO  → RMSE: {metricas['rmse_train']:.4f} | MAE: {metricas['mae_train']:.4f}")
        print(
            f"      TESTE   → RMSE: {metricas['rmse_test']:.4f} | MAE: {metricas['mae_test']:.4f}")

    except Exception as e:
        print(f"  ⚠️  {nome_modelo} [{norm_nome}] - Erro: {str(e)}")
        metricas = {
            'rmse_train': np.nan, 'mae_train': np.nan,
            'rmse_test': np.nan, 'mae_test': np.nan,
            'sucesso': False
        }

    return metricas


# ============================================
# 5. Execução dos Experimentos
# ============================================
n_rodadas = 1
resultados_completos = []

print("🔬 Iniciando experimentos...\n")

for rodada in range(n_rodadas):
    print(f"\n{'='*70}")
    print(f"RODADA {rodada + 1}/{n_rodadas}")
    print(f"{'='*70}")

    # Embaralhar e dividir
    seed = 20000+rodada
    np.random.seed(seed)
    indices = np.random.permutation(len(x))
    x_emb = x[indices]
    y_emb = y[indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x_emb, y_emb, test_size=0.2, random_state=seed
    )

    resultado_rodada = {'rodada': rodada + 1}

    # Loop por normalizador
    for norm_config in normalizadores:
        print(f"\n{'─'*70}")
        print(f"🔄 NORMALIZADOR: {norm_config['nome']}")
        print(f"{'─'*70}")

        scaler_x = norm_config['scaler_class']()
        scaler_y = norm_config['scaler_class']()

        x_train_norm = scaler_x.fit_transform(x_train)
        x_test_norm = scaler_x.transform(x_test)
        y_train_norm = scaler_y.fit_transform(y_train)
        y_test_norm = scaler_y.transform(y_test)

        norm_nome = norm_config['nome']

        # Testar MLPs
        print("\n  🧠 REDES NEURAIS (MLP)")
        for config in mlp_configs:
            mlp = criar_mlp(
                config['activation'], config['learning_rate'], rodada, config['verbose'])
            metricas = avaliar_modelo(mlp, x_train_norm, x_test_norm,
                                      y_train_norm, y_test_norm, scaler_y,
                                      config['name'], norm_nome)

            for nome_metrica, valor in metricas.items():
                if nome_metrica != 'sucesso':
                    metrica_tipo = nome_metrica.split('_')[0].upper()
                    conjunto = 'Treino' if 'train' in nome_metrica else 'Teste'
                    col_name = f"{metrica_tipo}_{config['name']}_{norm_nome}_{conjunto}"
                    resultado_rodada[col_name] = valor

        # Testar SVMs
        print("\n  🎯 SUPPORT VECTOR MACHINES (SVM)")
        for config in svm_configs:
            svm = criar_svm(config['kernel'], config['C'], config['epsilon'],
                            config.get('degree', 2))
            metricas = avaliar_modelo(svm, x_train_norm, x_test_norm,
                                      y_train_norm, y_test_norm, scaler_y,
                                      config['name'], norm_nome)

            for nome_metrica, valor in metricas.items():
                if nome_metrica != 'sucesso':
                    metrica_tipo = nome_metrica.split('_')[0].upper()
                    conjunto = 'Treino' if 'train' in nome_metrica else 'Teste'
                    col_name = f"{metrica_tipo}_{config['name']}_{norm_nome}_{conjunto}"
                    resultado_rodada[col_name] = valor

        # Testar RLs
        print("\n  📈 REGRESSÃO LINEAR")
        for config in rl_configs:
            rl = criar_reglinear(config['penalty'], config['alpha'])
            metricas = avaliar_modelo(rl, x_train_norm, x_test_norm,
                                      y_train_norm, y_test_norm, scaler_y,
                                      config['name'], norm_nome)

            for nome_metrica, valor in metricas.items():
                if nome_metrica != 'sucesso':
                    metrica_tipo = nome_metrica.split('_')[0].upper()
                    conjunto = 'Treino' if 'train' in nome_metrica else 'Teste'
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
                media_treino = df_resultados[col_treino].mean()
                std_treino = df_resultados[col_treino].std()
                linha[f"{metrica}_Teste_Média"] = media_teste
                linha[f"{metrica}_Teste_DP"] = std_teste
                linha[f"{metrica}_Treino_Média"] = media_treino
                linha[f"{metrica}_Treino_DP"] = std_treino
        tabela_resumo.append(linha)

    tabelas_por_norm[norm_nome] = pd.DataFrame(tabela_resumo)

df_resumo_completo = pd.concat(tabelas_por_norm.values(), ignore_index=True)
df_resumo_completo.to_csv("tabela_resumo_normalizacoes.csv",
                          index=False, sep=';', decimal=',')
print("💾 Arquivo salvo: tabela_resumo_normalizacoes.csv")



# FIGURA 1: Ranking Geral
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 3.8))
fig2.suptitle('Figura 2 - Ranking Geral por Normalização',
              fontsize=16, fontweight='bold')

ranking_std = tabelas_por_norm['Standard'].sort_values('RMSE_Teste_Média')
colors1 = plt.cm.viridis(np.linspace(0.2, 0.9, len(ranking_std)))
ax1.barh(range(len(ranking_std)),
         ranking_std['RMSE_Teste_Média'], color=colors1)
ax1.set_yticks(range(len(ranking_std)))
ax1.set_yticklabels(ranking_std['Modelo'], fontsize=9)
ax1.set_xlabel('RMSE Médio', fontsize=11)
ax1.set_title('StandardScaler', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(ranking_std.iterrows()):
    ax1.text(row['RMSE_Teste_Média'], i, f" {row['RMSE_Teste_Média']:.3f}",
             va='center', fontsize=8)

ranking_mm = tabelas_por_norm['MinMax'].sort_values('RMSE_Teste_Média')
colors2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(ranking_mm)))
ax2.barh(range(len(ranking_mm)), ranking_mm['RMSE_Teste_Média'], color=colors2)
ax2.set_yticks(range(len(ranking_mm)))
ax2.set_yticklabels(ranking_mm['Modelo'], fontsize=9)
ax2.set_xlabel('RMSE Médio', fontsize=11)
ax2.set_title('MinMaxScaler', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(ranking_mm.iterrows()):
    ax2.text(row['RMSE_Teste_Média'], i, f" {row['RMSE_Teste_Média']:.3f}",
             va='center', fontsize=8)

plt.tight_layout()
plt.savefig("figura1_ranking_normalizacoes.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura1_ranking_normalizacoes.png")

# FIGURA 3: Diferença Percentual
fig3, ax = plt.subplots(figsize=(6.3, 3.8))
fig3.suptitle('Figura 2 - Impacto da Normalização (% de melhoria com MinMax)',
              fontsize=14, fontweight='bold')

diferencas = []
for modelo in todos_modelos:
    std_row = tabelas_por_norm['Standard'][
        tabelas_por_norm['Standard']['Modelo'] == modelo.replace('_', ' ')
    ].iloc[0]
    mm_row = tabelas_por_norm['MinMax'][
        tabelas_por_norm['MinMax']['Modelo'] == modelo.replace('_', ' ')
    ].iloc[0]

    diff_pct = ((mm_row['RMSE_Teste_Média'] -
                std_row['RMSE_Teste_Média']) / std_row['RMSE_Teste_Média']) * 100
    diferencas.append((modelo.replace('_', ' '), diff_pct))

diferencas.sort(key=lambda x: x[1])
modelos_sorted = [d[0] for d in diferencas]
diffs_sorted = [d[1] for d in diferencas]

colors = ['green' if d < 0 else 'red' for d in diffs_sorted]
bars = ax.barh(range(len(modelos_sorted)),
               diffs_sorted, color=colors, alpha=0.7)
ax.set_yticks(range(len(modelos_sorted)))
ax.set_yticklabels(modelos_sorted, fontsize=9)
ax.set_xlabel('Diferença % (negativo = MinMax melhor)', fontsize=11)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

for i, (modelo, diff) in enumerate(diferencas):
    ax.text(diff, i, f' {diff:+.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("figura2_impacto_normalizacao.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura2_impacto_normalizacao.png")

print("\n" + "="*70)
print("✅ ANÁLISE DE NORMALIZAÇÃO FINALIZADA!")
print("="*70)
print(f"\nTempo de execução: {datetime.now().strftime('%H:%M:%S')}")
print("\n📁 Arquivos gerados:")
print("   1. resultados_dual_normalizacao.csv")
print("   2. tabela_resumo_normalizacoes.csv")
print("   3. figura1_ranking_normalizacoes.png")
print("   4. figura2_impacto_normalizacao.png")
print("="*70)

# ============================================
# VISUALIZAÇÕES ADICIONAIS PARA O TCC
# ============================================

print("\n" + "="*70)
print("📊 GERANDO VISUALIZAÇÕES ADICIONAIS PARA ANÁLISE")
print("="*70 + "\n")

# ============================================
# FIGURA 3: TOP 10 MODELOS (GERAL)
# ============================================
print("📈 Gerando Top 10 Modelos...")

# Coletar todos os modelos com suas métricas
todos_resultados = []
for norm_config in normalizadores:
    norm_nome = norm_config['nome']
    for modelo in todos_modelos:
        row = tabelas_por_norm[norm_nome][
            tabelas_por_norm[norm_nome]['Modelo'] == modelo.replace('_', ' ')
        ].iloc[0]

        todos_resultados.append({
            'Modelo': f"{modelo.replace('_', ' ')} [{norm_nome}]",
            'RMSE_Media': row['RMSE_Teste_Média'],
            'RMSE_DP': row['RMSE_Teste_DP'],
            'MAE_Media': row['MAE_Teste_Média']
        })

df_todos = pd.DataFrame(todos_resultados).sort_values('RMSE_Media').head(10)

# ============================================
# FIGURA 3: ANÁLISE DE ESTABILIDADE (CV)
# ============================================
print("📈 Gerando análise de estabilidade...")

fig7, axes = plt.subplots(1, 2, figsize=(6.3, 3.8))
fig7.suptitle('Figura 3 - Análise de Estabilidade (Coeficiente de Variação)',
              fontsize=14, fontweight='bold')

for idx, norm_config in enumerate(normalizadores):
    norm_nome = norm_config['nome']
    ax = axes[idx]

    cvs = []
    labels = []

    for modelo in todos_modelos:
        row = tabelas_por_norm[norm_nome][
            tabelas_por_norm[norm_nome]['Modelo'] == modelo.replace('_', ' ')
        ].iloc[0]

        cv = (row['RMSE_Teste_DP'] / row['RMSE_Teste_Média']) * 100
        cvs.append(cv)
        labels.append(modelo.replace('_', ' '))

    # Ordenar por CV
    sorted_indices = np.argsort(cvs)
    cvs_sorted = [cvs[i] for i in sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cvs_sorted)))

    bars = ax.barh(range(len(cvs_sorted)), cvs_sorted, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=9)
    ax.set_xlabel('Coeficiente de Variação (%)', fontsize=11)
    ax.set_title(f'{norm_nome}Scaler', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    for i, cv in enumerate(cvs_sorted):
        ax.text(cv + 0.2, i, f'{cv:.2f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("figura3_estabilidade_modelos.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura3_estabilidade_modelos.png")


# ============================================
# FIGURA 4: COMPARAÇÃO DE HIPERPARÂMETROS - MLP
# ============================================
print("📈 Gerando comparação de hiperparâmetros MLP...")

fig9, axes = plt.subplots(1, 2, figsize=(6.3, 3.8))
fig9.suptitle('Figura 4 - Impacto de Hiperparâmetros: MLP',
              fontsize=14, fontweight='bold')

for idx, norm_config in enumerate(normalizadores):
    norm_nome = norm_config['nome']
    ax = axes[idx]

    # Organizar dados por função de ativação e lr
    ativacoes = ['tanh', 'sigmoid', 'relu']
    lrs = [0.005, 0.01]

    dados_plot = {ativ: [] for ativ in ativacoes}

    for ativ in ativacoes:
        for lr in lrs:
            modelo_nome = f"MLP_{ativ}_lr_{lr}"
            row = tabelas_por_norm[norm_nome][
                tabelas_por_norm[norm_nome]['Modelo'].str.contains(ativ, case=False) &
                tabelas_por_norm[norm_nome]['Modelo'].str.contains(str(lr))
            ]
            if not row.empty:
                dados_plot[ativ].append(row.iloc[0]['RMSE_Teste_Média'])

    x = np.arange(len(lrs))
    width = 0.25

    for i, ativ in enumerate(ativacoes):
        offset = width * (i - 1)
        ax.bar(x + offset, dados_plot[ativ], width,
               label=ativ.capitalize(), alpha=0.8)

    ax.set_xlabel('Taxa de Aprendizado', fontsize=11)
    ax.set_ylabel('RMSE Teste Médio', fontsize=11)
    ax.set_title(f'{norm_nome}Scaler', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['0.005', '0.01'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("figura4_hiperparametros_mlp.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura4_hiperparametros_mlp.png")

# ============================================
# FIGURA 5: COMPARAÇÃO DE KERNELS - SVM
# ============================================
print("📈 Gerando comparação de kernels SVM...")

fig10, axes = plt.subplots(1, 2, figsize=(6.3, 3.8))
fig10.suptitle('Figura 5 - Impacto de Hiperparâmetros: SVM',
               fontsize=14, fontweight='bold')

for idx, norm_config in enumerate(normalizadores):
    norm_nome = norm_config['nome']
    ax = axes[idx]

    kernels = ['rbf', 'linear', 'poly']
    e = [0.1, 0.3]

    dados_plot = {kernel: [] for kernel in kernels}

    for kernel in kernels:
        for epsilon in e:
            modelo_nome = f"SVM_{kernel}_epsilon_{e}"
            row = tabelas_por_norm[norm_nome][
                tabelas_por_norm[norm_nome]['Modelo'].str.contains(kernel, case=False) &
                tabelas_por_norm[norm_nome]['Modelo'].str.contains(f'epsilon {epsilon}')
            ]
            if not row.empty:
                dados_plot[kernel].append(row.iloc[0]['RMSE_Teste_Média'])

    x = np.arange(len(e))
    width = 0.25

    for i, kernel in enumerate(kernels):
        offset = width * (i - 1)
        ax.bar(x + offset, dados_plot[kernel], width,
               label=kernel.upper(), alpha=0.8)

    ax.set_xlabel('Epsilon', fontsize=11)
    ax.set_ylabel('RMSE Teste Médio', fontsize=11)
    ax.set_title(f'{norm_nome}Scaler', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['epsilon=0.1', 'epsilon=0.3'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("figura5_hiperparametros_svm.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura5_hiperparametros_svm.png")

# ============================================
# FIGURA 6: COMPARAÇÃO REGULARIZAÇÃO - RL
# ============================================
print("📈 Gerando comparação de regularização RL...")

fig11, axes = plt.subplots(1, 2, figsize=(6.3, 3.8))
fig11.suptitle('Figura 6 - Impacto de Regularização: Regressão Linear',
               fontsize=14, fontweight='bold')

for idx, norm_config in enumerate(normalizadores):
    norm_nome = norm_config['nome']
    ax = axes[idx]

    modelos_rl = ['RL MQO', 'RL Ridge alpha 0.1', 'RL Ridge alpha 10',
                  'RL Lasso alpha 0.1', 'RL Lasso alpha 0.5', 'RL Lasso alpha 1']

    valores = []
    labels_curtos = []

    for modelo in modelos_rl:
        row = tabelas_por_norm[norm_nome][
            tabelas_por_norm[norm_nome]['Modelo'] == modelo
        ]
        if not row.empty:
            valores.append(row.iloc[0]['RMSE_Teste_Média'])
            # Criar labels mais curtos
            if 'MQO' in modelo:
                labels_curtos.append('MQO')
            elif 'Ridge' in modelo:
                alpha = modelo.split('alpha ')[-1]
                labels_curtos.append(f'Ridge\nα={alpha}')
            elif 'Lasso' in modelo:
                alpha = modelo.split('alpha ')[-1]
                labels_curtos.append(f'Lasso\nα={alpha}')

    colors = ['gray'] + ['blue']*2 + ['red']*3
    bars = ax.bar(range(len(valores)), valores, color=colors, alpha=0.7)

    ax.set_xticks(range(len(labels_curtos)))
    ax.set_xticklabels(labels_curtos, fontsize=9)
    ax.set_ylabel('RMSE Teste Médio', fontsize=11)
    ax.set_title(f'{norm_nome}Scaler', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(valores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("figura6_regularizacao_rl.png", dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white')
print("✅ Salvo: figura6_regularizacao_rl.png")


# ============================================
# RELATÓRIO FINAL
# ============================================
print("\n" + "="*70)
print("✅ TODAS AS VISUALIZAÇÕES ADICIONAIS GERADAS!")
print("="*70)
print("\n📁 Novos arquivos criados:")
print("   5. figura4_estabilidade_modelos.png - Coeficiente de variação")
print("   6. figura5_hiperparametros_mlp.png - Ativação e learning rate")
print("  7. figura6_hiperparametros_svm.png - Kernels e parâmetro C")
print("  8. figura7_regularizacao_rl.png - Ridge vs Lasso")
print("="*70)
