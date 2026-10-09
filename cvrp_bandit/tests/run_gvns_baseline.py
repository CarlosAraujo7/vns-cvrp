#!/usr/bin/env python3
"""
run_gvns_baseline.py
=====================
Roda o GVNS baseline (sem bandit -- ordem de vizinhança fixa, ver
neighborhoods.NEIGHBORHOODS) em todas as instâncias de um conjunto
(A, B, X ou CMT), com múltiplas seeds, e compara com a coluna "VNS mean
error" das Tabelas 10/11/12 (ou "Heuristic/GVNS" da Tabela 13, para CMT)
de Kalatzantonakis et al. (2023).

Protocolo adotado (documentado explicitamente porque o artigo não deixa
100% claro qual foi usado na comparação final das Tabelas 10-12 -- só
descreve que 5/10/20 steps foram testados durante o ajuste de
hiperparâmetro, Tabela 8): steps_max=20 (o maior valor testado no artigo),
5 seeds por instância, reportando erro melhor-de-5 e médio-de-5. Isso é
uma REPRODUÇÃO DE BOA-FÉ, não uma réplica garantida byte-a-byte (nem
poderia ser, sem o código/seeds originais dos autores) -- o objetivo desta
etapa é validar que a implementação está correta e competitiva, não bater
os números com precisão de casas decimais.

Uso:
    python run_gvns_baseline.py A
    python run_gvns_baseline.py B
    python run_gvns_baseline.py X
    python run_gvns_baseline.py CMT
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
from vns import gvns  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000

# "VNS mean error" reportado nas Tabelas 10, 11, 12 do artigo (médias sobre
# as instâncias do conjunto); para CMT usamos a linha "Heuristic/LKH3" IGUAL
# não -- CMT na Tabela 13 não tem uma linha "GVNS clássico" direta, então
# CMT fica de fora desta comparação agregada (só reportamos nosso próprio
# resultado; comparação com a Tabela 13 fica para quando o bandit entrar).
PAPER_VNS_MEAN_ERROR = {"A": 2.0401, "B": 1.8324, "X": 6.2906}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("A", "B", "X", "CMT"):
        print("uso: python run_gvns_baseline.py {A|B|X|CMT}")
        sys.exit(2)
    set_name = sys.argv[1]
    inst_dir = ROOT / "instances" / set_name
    vrp_files = sorted(inst_dir.glob("*.vrp"))
    # vehicles_exact e' autodetectado por instancia a partir do NOME (ver
    # instance.py, _vehicles_exact_default): X e CMT usam K como limite
    # INFERIOR; A e B exigem EXATAMENTE K (bug do CMT corrigido em
    # 29/08/2026, diario Entrada 3 -- Uchoa et al. 2017 confirma que a
    # re-publicacao CMT das instancias E/M tambem abandonou o K fixo, nao
    # so' o arredondamento de distancia).

    out_csv = ROOT / "results" / f"gvns_baseline_{set_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    linhas = []
    t_inicio_total = time.perf_counter()
    for vrp_path in vrp_files:
        inst = load_instance(vrp_path)
        # reparo (CWS + repair_route_count) e' DETERMINISTICO -- nao depende
        # da seed -- entao e' calculado uma unica vez por instancia, fora do
        # loop de seeds. Bug de eficiencia corrigido em 29/08/2026 (diario,
        # Entrada 3): a versao anterior recalculava dentro do loop de seeds,
        # o que re-resolvia o fallback CP-SAT (ate' 60s) 5x por instancia
        # desnecessariamente para as instancias que precisam dele.
        t0_reparo = time.perf_counter()
        # CORRECAO (diario, Entrada 88, achado F1): o comentario acima estava ERRADO -- o reparo
        # com fallback CP-SAT (limite de tempo, varios threads) NAO e deterministico em 11
        # instancias A/B. A solucao inicial agora e FIXADA em disco (src/solucao_inicial.py).
        cws = obter_inicial(inst)
        dt_reparo = time.perf_counter() - t0_reparo
        erros = []
        tempos = []
        for s in range(N_SEEDS):
            seed = SEED_BASE + s
            rng = random.Random(seed)
            t0 = time.perf_counter()
            resultado = gvns(inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX)
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
                "tempo_reparo_s": dt_reparo,
            })
        melhor = min(erros)
        media = sum(erros) / len(erros)
        marca_reparo = f"  reparo={dt_reparo:5.1f}s" if dt_reparo > 1.0 else ""
        print(
            f"{inst.name:14s} n={inst.n_customers:4d}  "
            f"erro_melhor_de_{N_SEEDS}={melhor:6.2f}%  erro_medio={media:6.2f}%  "
            f"tempo_medio={sum(tempos)/len(tempos):6.2f}s{marca_reparo}"
        )

    dt_total = time.perf_counter() - t_inicio_total

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    # agregados
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

    print(f"\n=== Conjunto {set_name}: {len(vrp_files)} instâncias, "
          f"{N_SEEDS} seeds cada, steps_max={STEPS_MAX} ===")
    print(f"Tempo total de execução: {dt_total:.1f}s")
    print(f"Erro médio (melhor-de-{N_SEEDS} por instância, média sobre instâncias): "
          f"{media_dos_melhores:.4f}%")
    print(f"Erro médio (média-de-{N_SEEDS} por instância, média sobre instâncias): "
          f"{media_das_medias:.4f}%")
    if set_name in PAPER_VNS_MEAN_ERROR:
        alvo = PAPER_VNS_MEAN_ERROR[set_name]
        print(f"'VNS mean error' reportado no artigo para o conjunto {set_name}: {alvo:.4f}%")
        print(f"Diferença (melhor-de-{N_SEEDS} vs. artigo): {media_dos_melhores - alvo:+.4f} p.p.")
        print(f"Diferença (média-de-{N_SEEDS} vs. artigo): {media_das_medias - alvo:+.4f} p.p.")
    print(f"\nCSV salvo em: {out_csv}")


if __name__ == "__main__":
    main()
