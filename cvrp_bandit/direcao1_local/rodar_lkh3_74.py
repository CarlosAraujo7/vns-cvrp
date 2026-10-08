#!/usr/bin/env python3
"""
rodar_lkh3_74.py
===================
Roda `src/solver_lkh3.py` (já validado -- diário, Entrada 36, inclusive a
armadilha do arredondamento interno de distância do LKH3 para CMT, já
neutralizada dentro do wrapper) nas 74 instâncias completas, com o mesmo
protocolo de orçamento de tempo igualado ao HG-UCB e a mesma simetria de
5 seeds/restarts que `rodar_ortools_74.py` -- ver a docstring de lá para
os detalhes do protocolo (é o mesmo, só troca o solver).

PRÉ-REQUISITO: `setup.sh` precisa ter baixado e compilado o LKH3 em
`external/LKH-3.0.7/LKH` (é onde `src/solver_lkh3.py` procura o binário,
caminho fixo relativo à raiz do projeto -- ver `LKH_BIN` naquele arquivo).

Uso: python rodar_lkh3_74.py
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instance import load_all  # noqa: E402
from solver_lkh3 import solve_lkh3, LKH_BIN  # noqa: E402

N_SEEDS = 5
SEED_BASE = 1000
CONJUNTOS = ("A", "B", "X", "CMT")

TEMPOS_CSV = ROOT / "results" / "tempos_hgucb_74.csv"
OUT_CSV = ROOT / "results" / "lkh3_74.csv"

CAMPOS = ["conjunto", "instancia", "n_customers", "seed", "time_limit_s", "tempo_gasto_s",
          "n_rotas", "n_vehicles_ref", "custo", "otimo", "gap_pct", "viavel", "motivo"]


def carregar_orcamentos_de_tempo():
    if not TEMPOS_CSV.exists():
        print(f"ERRO: {TEMPOS_CSV} não existe -- rode medir_tempo_hgucb.py primeiro.")
        sys.exit(1)
    soma, conta = {}, {}
    with open(TEMPOS_CSV, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            nome = linha["instancia"]
            soma[nome] = soma.get(nome, 0.0) + float(linha["tempo_s"])
            conta[nome] = conta.get(nome, 0) + 1
    return {nome: soma[nome] / conta[nome] for nome in soma}


def main():
    if not LKH_BIN.exists():
        print(f"ERRO: binário do LKH3 não encontrado em {LKH_BIN} -- rode setup.sh primeiro.")
        sys.exit(1)

    orcamentos = carregar_orcamentos_de_tempo()
    print(f"=== Rodando LKH3 nas 74 instâncias (orçamento de tempo = média do HG-UCB por instância) ===\n")

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
                print(f"  AVISO: {inst.name} não tem tempo medido em {TEMPOS_CSV.name} -- pulando.")
                continue
            time_limit_s = orcamentos[inst.name]
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                try:
                    routes, tempo_gasto = solve_lkh3(inst, time_limit_s, seed=seed)
                except RuntimeError as e:
                    # LKH3 nao achou solucao viavel em NENHUM candidato de VEHICLES
                    # dentro do orcamento -- unico raise RuntimeError de solve_lkh3
                    # (ver solver_lkh3.py), nao um bug generico sendo escondido.
                    # CORRECAO (diario, Entrada 48): a suposicao original de que
                    # solve_lkh3 sinalizava esse caso retornando `None` (Entrada 46)
                    # estava ERRADA -- a funcao SEMPRE levanta RuntimeError quando
                    # nenhum candidato de VEHICLES acha solucao, nunca retorna None
                    # ao chamador; o `if routes is None` abaixo nunca disparava de
                    # fato. Mesmo tratamento de rodar_ortools_74.py: registra como
                    # inviavel e CONTINUA, em vez de derrubar a execucao toda.
                    print(f"    AVISO: {inst.name} seed={seed}: {e} -- registrando como inviável.")
                    writer.writerow({
                        "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
                        "seed": seed, "time_limit_s": time_limit_s, "tempo_gasto_s": "",
                        "n_rotas": 0, "n_vehicles_ref": inst.n_vehicles_ref,
                        "custo": "", "otimo": inst.optimum, "gap_pct": "",
                        "viavel": False, "motivo": "LKH3 nao encontrou solucao no orcamento",
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
