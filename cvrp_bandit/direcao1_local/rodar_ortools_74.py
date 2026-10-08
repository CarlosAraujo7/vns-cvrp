#!/usr/bin/env python3
"""
rodar_ortools_74.py
======================
Roda `src/solver_ortools.py` (já validado -- diário, Entrada 35) nas 74
instâncias completas (A=26/B=23/X=21/CMT=4), com o orçamento de tempo por
instância IGUALADO ao tempo de parede do nosso próprio HG-UCB (protocolo
aprovado, Entrada 34, decisão 3) -- lê `results/tempos_hgucb_74.csv`
(gerado por `medir_tempo_hgucb.py`, que precisa rodar ANTES deste
script), usa a MÉDIA dos 5 seeds do HG-UCB como orçamento de tempo para
aquela instância (decisão de design documentada no diário: a média evita
que o orçamento do comparador fique refém do ruído de uma única
execução).

Mesma simetria de seeds do nosso próprio protocolo -- 5 seeds/restarts
por instância (SEED_BASE=1000), não uma única execução (decisão autônoma
já documentada, Entrada 34: OR-Tools e LKH3 têm componentes de
aleatoriedade internos, uma única execução de cada seria menos robusta
que a nossa comparação de 5 seeds).

Salva CSV incrementalmente (uma linha por escrita, flush a cada
instância) -- como esta execução pode levar horas, cada instância
concluída fica salva no disco mesmo que o processo seja interrompido
antes do fim.

Uso: python rodar_ortools_74.py
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instance import load_all  # noqa: E402
from solver_ortools import solve_ortools  # noqa: E402

N_SEEDS = 5
SEED_BASE = 1000
CONJUNTOS = ("A", "B", "X", "CMT")

TEMPOS_CSV = ROOT / "results" / "tempos_hgucb_74.csv"
OUT_CSV = ROOT / "results" / "ortools_74.csv"

CAMPOS = ["conjunto", "instancia", "n_customers", "seed", "time_limit_s", "tempo_gasto_s",
          "n_rotas", "n_vehicles_ref", "custo", "otimo", "gap_pct", "viavel", "motivo"]


def carregar_orcamentos_de_tempo():
    if not TEMPOS_CSV.exists():
        print(f"ERRO: {TEMPOS_CSV} não existe -- rode medir_tempo_hgucb.py primeiro "
              f"(o orçamento de tempo deste script vem de lá).")
        sys.exit(1)
    soma, conta = {}, {}
    with open(TEMPOS_CSV, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            nome = linha["instancia"]
            soma[nome] = soma.get(nome, 0.0) + float(linha["tempo_s"])
            conta[nome] = conta.get(nome, 0) + 1
    return {nome: soma[nome] / conta[nome] for nome in soma}


def main():
    orcamentos = carregar_orcamentos_de_tempo()
    print(f"=== Rodando OR-Tools nas 74 instâncias (orçamento de tempo = média do HG-UCB por instância) ===\n")

    ja_existe = OUT_CSV.exists()
    OUT_CSV.parent.mkdir(exist_ok=True)
    f_out = open(OUT_CSV, "a" if ja_existe else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=CAMPOS)
    if not ja_existe:
        writer.writeheader()

    t_inicio_total = time.perf_counter()
    for set_name in CONJUNTOS:
        instancias = load_all(ROOT / "instances" / set_name)
        for inst in instancias:
            if inst.name not in orcamentos:
                print(f"  AVISO: {inst.name} não tem tempo medido em {TEMPOS_CSV.name} -- pulando "
                      f"(rode medir_tempo_hgucb.py de novo se essa instância foi adicionada depois).")
                continue
            time_limit_s = orcamentos[inst.name]
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                try:
                    routes, tempo_gasto = solve_ortools(inst, time_limit_s, seed=seed)
                except RuntimeError as e:
                    # OR-Tools nao achou NENHUMA solucao dentro do orcamento de tempo/config
                    # de veiculos (caso especifico e ja esperado -- ver solver_ortools.py,
                    # unico raise RuntimeError da funcao -- nao um bug generico sendo
                    # escondido). Mesmo tratamento ja usado em rodar_lkh3_74.py: registra
                    # como inviavel e CONTINUA, em vez de derrubar a execucao toda.
                    print(f"    AVISO: {inst.name} seed={seed}: {e} -- registrando como inviável.")
                    writer.writerow({
                        "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
                        "seed": seed, "time_limit_s": time_limit_s, "tempo_gasto_s": "",
                        "n_rotas": 0, "n_vehicles_ref": inst.n_vehicles_ref,
                        "custo": "", "otimo": inst.optimum, "gap_pct": "",
                        "viavel": False, "motivo": "OR-Tools nao encontrou solucao no orcamento",
                    })
                    continue
                custo = inst.solution_cost(routes)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                viavel, motivo = inst.is_feasible(routes)
                gaps.append(gap)
                writer.writerow({
                    "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
                    "seed": seed, "time_limit_s": time_limit_s, "tempo_gasto_s": tempo_gasto,
                    "n_rotas": len([r for r in routes if r]), "n_vehicles_ref": inst.n_vehicles_ref,
                    "custo": custo, "otimo": inst.optimum, "gap_pct": gap,
                    "viavel": viavel, "motivo": motivo,
                })
            f_out.flush()
            media_g = sum(gaps) / len(gaps) if gaps else float("nan")
            print(f"  {inst.name:16s} n={inst.n_customers:4d}  orcamento={time_limit_s:7.3f}s  "
                  f"gap_medio={media_g:6.2f}%  ({len(gaps)}/{N_SEEDS} seeds viaveis)")

    f_out.close()
    dt_total = time.perf_counter() - t_inicio_total
    print(f"\nTempo total de execução deste script: {dt_total:.1f}s ({dt_total/60:.1f}min)")
    print(f"CSV salvo em: {OUT_CSV}")


if __name__ == "__main__":
    main()
