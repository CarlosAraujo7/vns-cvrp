#!/usr/bin/env python3
"""
retry_lkh3_piso_maior.py
===========================
Re-tenta, com um piso de tempo bem maior, as instâncias do LKH3 que
ficaram com ZERO seeds viáveis em `results/lkh3_74.csv` (rodado por
`rodar_lkh3_74.py`) -- decisão do pesquisador (via `AskUserQuestion`,
respondendo ao diário Entrada 50): "tentar mesmo assim, aceitando
resultado parcial".

POR QUE ISSO EXISTE (diário, Entradas 49/50 -- achado real #37): 8
instâncias (1 B + 7 X) ficaram sem NENHUMA solução viável no protocolo
de tempo igualado ao HG-UCB. Investigação no sandbox mostrou que o
problema NÃO é uniforme: `X-n157-k13` ficou viável dando 60s por
candidato de K (contra ~1,35s no protocolo normal), mas `X-n172-k51`
continuou inviável mesmo com 60s, 4 sementes diferentes e múltiplos
reinícios (RUNS=3) -- ou seja, ESTE SCRIPT NÃO GARANTE resolver todas as
8 instâncias. Espera-se um resultado PARCIAL: algumas serão resgatadas,
outras continuarão "sem solução" mesmo com o piso maior -- isso é
esperado e faz parte do achado, não um sinal de bug.

O QUE MUDA EM RELAÇÃO AO PROTOCOLO ORIGINAL: as instâncias tratadas por
este script deixam de seguir o protocolo de tempo IGUALADO ao HG-UCB
(diário Entrada 34) -- por isso elas são marcadas com um `motivo`
diferente ("ok (retry com piso de Ns/candidato)" em vez de "ok") no CSV
final, para que a análise final e a escrita do artigo possam tratar
esses pontos de dado de forma transparente, não escondida (mesmo
princípio de "exclusão reportada" já usado para os casos sem solução).

SEGURANÇA DO CSV: nunca sobrescreve `results/lkh3_74.csv` sem backup --
antes de qualquer alteração, salva uma cópia com timestamp
(`lkh3_74_pre_retry_AAAAMMDD_HHMMSS.csv`) do estado atual (protocolo de
tempo igualado puro, sem retry). Só as linhas das instâncias
efetivamente re-tentadas são alteradas; todo o resto do CSV permanece
byte-a-byte idêntico.

CUSTO DE TEMPO: este script IMPRIME uma estimativa de pior caso antes de
começar (ver a conta em `main()`) -- pode facilmente passar de 1-3h para
todas as 8 instâncias no piso padrão de 60s, já que instâncias X/CMT
(vehicles_exact=False) testam até 6 valores de K, cada um recebendo o
piso inteiro. Use `--somente` para testar 1-2 instâncias primeiro antes
de comprometer o lote inteiro (mesma disciplina de sempre: barato antes
de caro).

Uso:
    python retry_lkh3_piso_maior.py                          # todas as instancias 5/5 sem solucao, piso=60s
    python retry_lkh3_piso_maior.py --somente X-n157-k13      # so uma instancia (teste barato primeiro)
    python retry_lkh3_piso_maior.py --piso_segundos 90        # piso maior (mais caro, pode resgatar mais)
"""
import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instance import load_all  # noqa: E402
from solver_lkh3 import solve_lkh3  # noqa: E402

CSV_PATH = ROOT / "results" / "lkh3_74.csv"
CAMPOS = ["conjunto", "instancia", "n_customers", "seed", "time_limit_s", "tempo_gasto_s",
          "n_rotas", "n_vehicles_ref", "custo", "otimo", "gap_pct", "viavel", "motivo"]

# vehicle_slack=5 é o padrão de solver_lkh3.py (mesma constante,
# VEHICLE_SLACK) -- 6 candidatos de K testados para vehicles_exact=False.
N_CANDIDATOS_SWEEP = 6


def main():
    parser = argparse.ArgumentParser(
        description="Re-tenta com piso de tempo maior as instâncias do LKH3 sem NENHUMA seed viável.")
    parser.add_argument("--piso_segundos", type=float, default=60.0,
                         help="Segundos por candidato de K (padrão 60 -- validado no sandbox para "
                              "resgatar X-n157-k13; nem toda instância vai resolver com isso, ver "
                              "diário Entrada 50).")
    parser.add_argument("--somente", nargs="*", default=None,
                         help="Rodar só estas instâncias (nomes exatos), em vez de todas as "
                              "detectadas com 5/5 seeds sem solução. Útil para testar 1-2 antes do lote todo.")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"ERRO: {CSV_PATH} não existe -- rode rodar_lkh3_74.py primeiro.")
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    por_instancia = {}
    for linha in linhas:
        por_instancia.setdefault(linha["instancia"], []).append(linha)

    candidatas = sorted(nome for nome, ls in por_instancia.items()
                         if len(ls) == 5 and all(l["viavel"] == "False" for l in ls))
    if args.somente:
        faltando = set(args.somente) - set(candidatas)
        if faltando:
            print(f"AVISO: {sorted(faltando)} não estão na lista de instâncias 5/5 sem solução do "
                  f"CSV atual (já viáveis, parcialmente viáveis, ou nome errado) -- ignorando.")
        candidatas = [n for n in candidatas if n in args.somente]

    if not candidatas:
        print("Nenhuma instância candidata (5/5 seeds sem solução) encontrada -- nada a fazer.")
        return

    todas_instancias = {}
    for conjunto in ("A", "B", "X", "CMT"):
        for inst in load_all(ROOT / "instances" / conjunto):
            todas_instancias[inst.name] = inst

    print(f"=== Retry do LKH3 com piso de tempo maior ({args.piso_segundos:.0f}s por candidato de K) ===")
    print(f"Instâncias candidatas (5/5 seeds sem solução no CSV atual): {len(candidatas)} -- "
          f"{', '.join(candidatas)}\n")

    custo_estimado_s = 0.0
    for nome in candidatas:
        inst = todas_instancias[nome]
        n_cand = 1 if inst.vehicles_exact else N_CANDIDATOS_SWEEP
        custo_estimado_s += n_cand * args.piso_segundos * 5  # 5 seeds cada
    print(f"Estimativa de tempo em PIOR CASO (nenhum candidato converge antes do fim do piso): "
          f"~{custo_estimado_s:.0f}s (~{custo_estimado_s/60:.0f}min, ~{custo_estimado_s/3600:.2f}h). "
          f"Pode ser bem menor na prática se algum candidato achar solução antes do piso acabar -- "
          f"mas trate este número como o que você está topando gastar.\n")

    backup_path = CSV_PATH.with_name(
        f"lkh3_74_pre_retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    shutil.copy2(CSV_PATH, backup_path)
    print(f"Backup do CSV original (protocolo de tempo igualado puro, sem retry) salvo em: "
          f"{backup_path.name}\n")

    resgatadas = 0
    ainda_sem_solucao = 0
    for nome in candidatas:
        inst = todas_instancias[nome]
        n_cand = 1 if inst.vehicles_exact else N_CANDIDATOS_SWEEP
        time_limit_total = n_cand * args.piso_segundos
        print(f"--- {nome} (n={inst.n_customers}, vehicles_exact={inst.vehicles_exact}, "
              f"{n_cand} candidato(s) x {args.piso_segundos:.0f}s = até {time_limit_total:.0f}s/seed) ---")
        for linha in por_instancia[nome]:
            seed = int(linha["seed"])
            t0 = time.perf_counter()
            try:
                routes, tempo_gasto = solve_lkh3(inst, time_limit_total, seed=seed)
            except RuntimeError:
                dt = time.perf_counter() - t0
                print(f"    seed={seed}: AINDA sem solução ({dt:.1f}s)")
                linha["time_limit_s"] = time_limit_total
                linha["motivo"] = f"LKH3 sem solucao mesmo com piso de {args.piso_segundos:.0f}s/candidato"
                ainda_sem_solucao += 1
                continue
            custo = inst.solution_cost(routes)
            gap = 100 * (custo - inst.optimum) / inst.optimum
            viavel, motivo_feas = inst.is_feasible(routes)
            print(f"    seed={seed}: {'RESGATADA' if viavel else 'achou rota mas is_feasible=False'} "
                  f"em {tempo_gasto:.1f}s -- gap={gap:.2f}%")
            linha["time_limit_s"] = time_limit_total
            linha["tempo_gasto_s"] = tempo_gasto
            linha["n_rotas"] = len([r for r in routes if r])
            linha["custo"] = custo
            linha["gap_pct"] = gap
            linha["viavel"] = viavel
            linha["motivo"] = (f"ok (retry com piso de {args.piso_segundos:.0f}s/candidato)"
                                if viavel else motivo_feas)
            if viavel:
                resgatadas += 1
            else:
                ainda_sem_solucao += 1

            # salva incrementalmente apos CADA SEED (nao so apos cada instancia --
            # diferente das outras scripts deste diretorio, aqui cada seed pode
            # levar minutos, entao perder uma instancia inteira num Ctrl+C seria
            # muito mais caro que nos outros scripts). Reescreve o arquivo todo
            # (linhas ja tem as atualizacoes em memoria) -- 74 instancias x 5
            # linhas e' pequeno o bastante pra isso ser instantaneo.
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CAMPOS)
                writer.writeheader()
                writer.writerows(linhas)

    print(f"\n=== Resultado do retry ===")
    print(f"Seeds resgatados (agora viáveis): {resgatadas}")
    print(f"Seeds ainda sem solução mesmo com piso maior: {ainda_sem_solucao}")
    print(f"CSV atualizado em: {CSV_PATH}")
    print(f"Backup do original (protocolo puro, sem retry): {backup_path.name}")


if __name__ == "__main__":
    main()
