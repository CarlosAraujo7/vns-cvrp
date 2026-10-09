#!/usr/bin/env python3
"""
run_bandit_vns_validation.py
==============================
Tarefa #178: validação completa do Bandit VNS (HG-UCB, `bandit_vns()` em
vns.py) contra as colunas "HG-UCB mean error" das Tabelas 10 (Set A), 11
(Set B) e 12 (Set X), e a coluna "Gap (%)" da linha HG-UCB da Tabela 13
(CMT1/CMT2/CMT3/CMT11), de Kalatzantonakis et al. (2023).

Protocolo IDÊNTICO ao baseline clássico (tests/run_gvns_baseline.py,
diário Entrada 3): steps_max=20, 5 seeds por instância, reportando erro
melhor-de-5 e médio-de-5 -- mesma ressalva de lá: reprodução de boa-fé, não
réplica byte-a-byte (sem os seeds/código originais dos autores).

Usa p_hat_mode="q_values" (decisão da Entrada 6 -- experimento comparativo
mostrou empate exato contra "sliding_window" em 3 instâncias/3 seeds, então
mantivemos a leitura mais simples) e b_t=0.2 literal + max_inner_iter=40
como teto de segurança (decisão aprovada pelo pesquisador, Entrada 5, dado
que max(q_values)<b_t é matematicamente inatingível para k=5).

MODO (segundo argumento, opcional, default "fixo" -- Entrada 7):
  fixo         -- progressive_x_linha=False, leitura literal original (x'
                  fica fixo durante todo o loop interno de um passo). Achado
                  real da Entrada 7: desempenho muito abaixo do artigo em
                  A/B/CMT (Sets A/B/CMT: +2 a +4pp; Set X quase não mostrou
                  o efeito, +0.31pp). CSV: results/bandit_vns_validation_{set}.csv
                  (nomes SEM sufixo -- preserva os arquivos já produzidos
                  pela primeira rodada completa, nunca sobrescrever).
  progressivo  -- progressive_x_linha=True, x' avança para x'' a cada
                  melhora aceita dentro do mesmo passo externo (fix
                  diagnosticado na Entrada 7, validado em amostra reduzida
                  via tests/compare_progressive_variant.py). CSV:
                  results/bandit_vns_validation_progressivo_{set}.csv
                  (nome distinto -- as duas rodadas completas coexistem para
                  comparação lado a lado, nenhuma sobrescreve a outra).

RECOMPENSA (terceiro argumento, opcional, default "original" -- Entrada 12,
achado real #6): qual reward positivo (Eq. 12) e qual alpha usar no HG-UCB.
  original              -- reward_positive() tal como no artigo (rs = delta de
                            custo em unidades ABSOLUTAS) + alpha=5.0 (Tabela 8,
                            literal do artigo). É o default de sempre --
                            EXATAMENTE o comportamento já validado nas rodadas
                            fixo/progressivo (Entradas 7-8), preservado aqui
                            para retrocompatibilidade total (mesmos nomes de
                            CSV, sem sufixo extra).
  normalizado_alpha5.0  -- reward_positive_normalized() (NOSSA extensão, não
                            do artigo -- normaliza rs dividindo por sol_current,
                            tornando-o adimensional/relativo) + alpha=5.0
                            (mantém o hiperparâmetro do artigo intocado -- a
                            correção mais fiel ao artigo dentre as testadas,
                            já que não altera nenhum valor publicado). CSV:
                            results/bandit_vns_validation_{sufixo}normalizado_alpha5.0_{set}.csv
  normalizado_alpha0.5  -- reward_positive_normalized() + alpha=0.5 (ponto de
                            melhor desempenho observado na amostra do
                            diagnóstico 6, embora altere o hiperparâmetro).
                            CSV: results/bandit_vns_validation_{sufixo}normalizado_alpha0.5_{set}.csv

MAX_INNER_ITER (quarto argumento, opcional, default 40 -- Entrada 22, decisão
do pesquisador de ir direto à validação completa depois da sub-investigação
das Entradas 10-22, diagnósticos 9-13): teto do loop interno do Bandit VNS
(ver bandit_vns() em vns.py). Default 40 preserva EXATAMENTE o comportamento
e os nomes de CSV já sincronizados (retrocompatibilidade total -- omitir
este argumento é idêntico a chamar a versão anterior deste script). Qualquer
valor != 40 acrescenta um sufixo `mii{valor}_` ao nome do CSV, então nunca
sobrescreve os arquivos já produzidos. Motivação: diagnósticos 9-13 (amostra
de até 19/74 instâncias) mediram gap%/colapso do bandit em {20,40,80,160} x
{alpha=5,0, alpha=0,5}; mii=160 nunca piorou e reduziu o colapso do softmax
de forma monotônica (achado #12, 18/19 instâncias), mas a comparação entre
os dois alphas em mii=160 mostrou empate nos conjuntos A/B/X e vantagem de
alpha=0,5 concentrada e unânime no conjunto CMT (achado #14) -- daí a
validação completa nas 74 instâncias, para ver o efeito real quando CMT é
só 4/74=5% do benchmark, não 4/19=21% como na amostra dos diagnósticos.

Uso:
    python run_bandit_vns_validation.py A               # modo fixo, recompensa original (default)
    python run_bandit_vns_validation.py TODOS           # modo fixo, recompensa original (default)
    python run_bandit_vns_validation.py TODOS fixo       # equivalente ao acima, explícito
    python run_bandit_vns_validation.py TODOS progressivo
    python run_bandit_vns_validation.py X progressivo
    python run_bandit_vns_validation.py TODOS progressivo normalizado_alpha5.0
    python run_bandit_vns_validation.py TODOS progressivo normalizado_alpha0.5
    python run_bandit_vns_validation.py TODOS progressivo normalizado_alpha0.5 160  # Entrada 22
"""
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402,F401
from solucao_inicial import obter_inicial  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000  # mesmo SEED_BASE do baseline classico -- comparavel seed-a-seed
MAX_INNER_ITER = 40  # 8*k, k=5 -- ver bandit_vns() docstring / diario Entrada 5

# Configurações de "recompensa" -- ver docstring do módulo (RECOMPENSA) e
# diário Entrada 12 (achado real #6). "original" reproduz EXATAMENTE o
# comportamento já validado (reward_positive_fn=None -> reward_positive() em
# vns.py, alpha=5.0 = default de HGUCBBandit) -- retrocompatibilidade total.
CONFIG_RECOMPENSA = {
    "original": {"reward_positive_fn": None, "alpha": 5.0},
    "normalizado_alpha5.0": {"reward_positive_fn": reward_positive_normalized, "alpha": 5.0},
    "normalizado_alpha0.5": {"reward_positive_fn": reward_positive_normalized, "alpha": 0.5},
}

# "HG-UCB mean error" por instancia, Tabelas 10 (A), 11 (B), 12 (X) --
# extraidos de literatura/paper.txt. CMT usa a coluna "Gap (%)" da linha
# HG-UCB da Tabela 13 (so' 4 instancias: CMT1/CMT2/CMT3/CMT11).
PAPER_HGUCB_POR_INSTANCIA = {
    "A-n32-k5": 0.1892, "A-n33-k5": 0.0151, "A-n33-k6": 0.0403, "A-n34-k5": 0.1404,
    "A-n36-k5": 1.6628, "A-n37-k5": 0.0818, "A-n37-k6": 0.5833, "A-n38-k5": 0.6835,
    "A-n39-k5": 0.6090, "A-n39-k6": 0.8525, "A-n44-k6": 1.5653, "A-n45-k6": 2.3560,
    "A-n45-k7": 1.3565, "A-n46-k7": 0.8496, "A-n48-k7": 2.5673, "A-n53-k7": 2.2642,
    "A-n54-k7": 1.4533, "A-n55-k9": 1.2720, "A-n60-k9": 2.5366, "A-n61-k9": 2.2237,
    "A-n62-k8": 3.2572, "A-n63-k10": 2.6193, "A-n63-k9": 2.4863, "A-n64-k9": 4.0859,
    "A-n65-k9": 2.8123, "A-n69-k9": 3.4536,
    "B-n31-k5": 0.2592, "B-n34-k5": 0.1390, "B-n35-k5": 0.4572, "B-n38-k6": 0.2970,
    "B-n39-k5": 0.6574, "B-n41-k6": 0.9000, "B-n43-k6": 0.7881, "B-n44-k7": 1.2748,
    "B-n45-k5": 0.7189, "B-n45-k6": 3.9066, "B-n50-k7": 0.1745, "B-n50-k8": 1.1188,
    "B-n51-k7": 0.0, "B-n52-k7": 1.3117, "B-n56-k7": 1.0898, "B-n57-k7": 0.0,
    "B-n57-k9": 1.5955, "B-n63-k10": 4.0804, "B-n64-k9": 3.7605, "B-n66-k9": 2.3453,
    "B-n67-k10": 4.1199, "B-n68-k9": 3.0771, "B-n78-k10": 6.0025,
    "X-n101-k25": 3.5419, "X-n106-k14": 2.5488, "X-n110-k13": 4.4541, "X-n115-k10": 6.5690,
    "X-n120-k6": 6.1007, "X-n125-k30": 3.4693, "X-n129-k18": 7.2141, "X-n134-k13": 7.3767,
    "X-n139-k10": 9.1838, "X-n143-k7": 11.3161, "X-n148-k46": 3.5120, "X-n153-k22": 4.1616,
    "X-n157-k13": 3.6606, "X-n162-k11": 9.1788, "X-n167-k10": 10.6379, "X-n172-k51": 3.8806,
    "X-n176-k26": 5.2610, "X-n181-k23": 3.3869, "X-n186-k15": 9.5608, "X-n190-k8": 6.5027,
    "X-n195-k51": 4.9670,
    "CMT1": 0.0, "CMT2": 2.33, "CMT3": 2.96, "CMT11": 11.72,
}
PAPER_HGUCB_MEDIA_CONJUNTO = {"A": 1.6103, "B": 1.6554, "X": 6.0230, "CMT": 4.0025}


def rodar_conjunto(set_name, progressive=False, config_recompensa="original",
                    max_inner_iter=MAX_INNER_ITER):
    inst_dir = ROOT / "instances" / set_name
    vrp_files = sorted(inst_dir.glob("*.vrp"))
    if not vrp_files:
        print(f"AVISO: nenhuma instancia encontrada em {inst_dir}")
        return None

    cfg = CONFIG_RECOMPENSA[config_recompensa]
    reward_positive_fn = cfg["reward_positive_fn"]
    alpha = cfg["alpha"]

    sufixo = "progressivo_" if progressive else ""
    # "original" preserva EXATAMENTE os nomes já sincronizados (Entradas 7-8);
    # só as configs novas ganham um sufixo extra no nome do CSV.
    sufixo_cfg = "" if config_recompensa == "original" else f"{config_recompensa}_"
    # idem para max_inner_iter -- default (40) preserva os nomes já sincronizados,
    # qualquer outro valor (Entrada 22) ganha um sufixo próprio, nunca sobrescreve.
    sufixo_mii = "" if max_inner_iter == MAX_INNER_ITER else f"mii{max_inner_iter}_"
    out_csv = ROOT / "results" / f"bandit_vns_validation_{sufixo}{sufixo_cfg}{sufixo_mii}{set_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    linhas = []
    t_inicio_total = time.perf_counter()
    for vrp_path in vrp_files:
        inst = load_instance(vrp_path)
        t0_reparo = time.perf_counter()
        cws = obter_inicial(inst)   # solucao inicial FIXADA (diario, Entrada 88, achado F1)
        dt_reparo = time.perf_counter() - t0_reparo

        erros = []
        tempos = []
        for s in range(N_SEEDS):
            seed = SEED_BASE + s
            rng = random.Random(seed)
            bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)
            t0 = time.perf_counter()
            resultado = bandit_vns(
                inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
                bandit=bandit_obj, max_inner_iter=max_inner_iter,
                progressive_x_linha=progressive,
                reward_positive_fn=reward_positive_fn,
            )
            dt = time.perf_counter() - t0
            viavel, motivo = inst.is_feasible(resultado)
            if not viavel:
                print(f"ERRO: {inst.name} seed={seed} produziu solução inviável: {motivo}")
                sys.exit(1)
            custo = inst.solution_cost(resultado)
            erro_pct = 100.0 * (custo - inst.optimum) / inst.optimum
            erros.append(erro_pct)
            tempos.append(dt)
            linhas.append({
                "conjunto": set_name, "instancia": inst.name, "seed": seed,
                "n_customers": inst.n_customers, "optimum": inst.optimum,
                "custo_obtido": custo, "erro_pct": erro_pct, "tempo_s": dt,
            })
        melhor = min(erros)
        media = sum(erros) / len(erros)
        alvo = PAPER_HGUCB_POR_INSTANCIA.get(inst.name)
        alvo_str = f"  artigo_HGUCB={alvo:6.2f}%  dif_media={media-alvo:+6.2f}pp" if alvo is not None else ""
        print(
            f"{inst.name:14s} n={inst.n_customers:4d}  "
            f"erro_melhor_de_{N_SEEDS}={melhor:6.2f}%  erro_medio={media:6.2f}%  "
            f"tempo_medio={sum(tempos)/len(tempos):6.2f}s{alvo_str}"
        )

    dt_total = time.perf_counter() - t_inicio_total

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    por_instancia_melhor = {}
    por_instancia_media = {}
    for row in linhas:
        nome = row["instancia"]
        por_instancia_melhor.setdefault(nome, []).append(row["erro_pct"])
    for nome, lst in por_instancia_melhor.items():
        por_instancia_media[nome] = sum(lst) / len(lst)
        por_instancia_melhor[nome] = min(lst)

    media_dos_melhores = sum(por_instancia_melhor.values()) / len(por_instancia_melhor)
    media_das_medias = sum(por_instancia_media.values()) / len(por_instancia_media)

    modo_label = "progressivo" if progressive else "fixo"
    mii_label = f", max_inner_iter={max_inner_iter}" if max_inner_iter != MAX_INNER_ITER else ""
    print(f"\n=== Conjunto {set_name} [modo={modo_label}, recompensa={config_recompensa}{mii_label}]: "
          f"{len(vrp_files)} instâncias, "
          f"{N_SEEDS} seeds cada, steps_max={STEPS_MAX} (Bandit VNS / HG-UCB) ===")
    print(f"Tempo total de execução: {dt_total:.1f}s")
    print(f"Erro médio (melhor-de-{N_SEEDS} por instância, média sobre instâncias): "
          f"{media_dos_melhores:.4f}%")
    print(f"Erro médio (média-de-{N_SEEDS} por instância, média sobre instâncias): "
          f"{media_das_medias:.4f}%")
    if set_name in PAPER_HGUCB_MEDIA_CONJUNTO:
        alvo = PAPER_HGUCB_MEDIA_CONJUNTO[set_name]
        print(f"'HG-UCB mean error' reportado no artigo para o conjunto {set_name}: {alvo:.4f}%")
        print(f"Diferença (melhor-de-{N_SEEDS} vs. artigo): {media_dos_melhores - alvo:+.4f} p.p.")
        print(f"Diferença (média-de-{N_SEEDS} vs. artigo): {media_das_medias - alvo:+.4f} p.p.")
    print(f"CSV salvo em: {out_csv}")
    return {"melhor": media_dos_melhores, "media": media_das_medias, "tempo_s": dt_total}


USO = ("uso: python run_bandit_vns_validation.py {A|B|X|CMT|TODOS} [fixo|progressivo] "
       "[" + "|".join(CONFIG_RECOMPENSA.keys()) + "] [max_inner_iter]")


def main():
    if len(sys.argv) not in (2, 3, 4, 5) or sys.argv[1] not in ("A", "B", "X", "CMT", "TODOS"):
        print(USO)
        sys.exit(2)
    alvo = sys.argv[1]
    modo = sys.argv[2] if len(sys.argv) >= 3 else "fixo"
    if modo not in ("fixo", "progressivo"):
        print(USO)
        sys.exit(2)
    config_recompensa = sys.argv[3] if len(sys.argv) >= 4 else "original"
    if config_recompensa not in CONFIG_RECOMPENSA:
        print(USO)
        sys.exit(2)
    max_inner_iter = int(sys.argv[4]) if len(sys.argv) == 5 else MAX_INNER_ITER
    progressive = (modo == "progressivo")
    conjuntos = ["A", "B", "X", "CMT"] if alvo == "TODOS" else [alvo]
    resumo = {}
    for c in conjuntos:
        mii_label = f", max_inner_iter={max_inner_iter}" if max_inner_iter != MAX_INNER_ITER else ""
        print(f"\n{'='*70}\nCONJUNTO {c} [modo={modo}, recompensa={config_recompensa}{mii_label}]\n{'='*70}")
        r = rodar_conjunto(c, progressive=progressive, config_recompensa=config_recompensa,
                            max_inner_iter=max_inner_iter)
        if r is not None:
            resumo[c] = r
    if len(resumo) > 1:
        print(f"\n{'='*70}\nRESUMO GERAL (Bandit VNS / HG-UCB, modo={modo}, "
              f"recompensa={config_recompensa}, vs. artigo)\n{'='*70}")
        for c, r in resumo.items():
            alvo_c = PAPER_HGUCB_MEDIA_CONJUNTO.get(c)
            print(f"{c:5s} melhor-de-{N_SEEDS}={r['melhor']:6.2f}%  media-de-{N_SEEDS}={r['media']:6.2f}%  "
                  f"artigo={alvo_c:6.2f}%  tempo={r['tempo_s']:7.1f}s")


if __name__ == "__main__":
    main()
