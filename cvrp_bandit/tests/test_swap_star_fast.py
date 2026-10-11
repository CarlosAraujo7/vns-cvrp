#!/usr/bin/env python3
"""
Validacao do SWAP* eficiente (src/neighborhoods_fast.py) contra o SWAP* ingenuo (neighborhoods_ext.swap_star_best):
  (1) mesmo MELHOR delta (tolerancia 1e-7) e mesmo resultado 'melhorou' em solucoes aleatorias de instancias reais;
  (2) o movimento devolvido reproduz o delta informado (custo novo - custo antigo), e a solucao e viavel;
  (3) o custo em avaliacoes de distancia cai (razao ingenuo/eficiente reportada).
Uso: python tests/test_swap_star_fast.py   (do diretorio cvrp_bandit)
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "estudo_aos"))
from neighborhoods_ext import swap_star_best  # noqa: E402
from neighborhoods_fast import swap_star_fast_best  # noqa: E402
from run_aos_study import _instancia  # noqa: E402
from run_aos_etapa3 import _proxy  # noqa: E402
from instance import load_all  # noqa: E402


def solucao_aleatoria(inst, rng):
    """parte da inicial gravada e embaralha clientes entre rotas respeitando capacidade (varios movimentos aleatorios)"""
    _, cws = _instancia(inst[0], inst[1])
    rotas = [list(r) for r in cws]
    return rotas


def custo(px, rotas):
    return px._inst.solution_cost([r for r in rotas if r])


def main():
    rng = random.Random(2026)
    tot, ok_improve, ratio_acc = 0, 0, []
    for conj, idx in (("SYN_TUNE", 0), ("SYN_TUNE", 9), ("SYN_TUNE", 15), ("SYN_TUNE", 23), ("A", 5), ("X", 3)):
        nomes = sorted(i.name for i in load_all(ROOT / "instances" / conj))
        nome = nomes[idx % len(nomes)]
        inst, cws = _instancia(conj, nome)
        px = _proxy(conj, nome)
        base = [list(r) for r in cws]
        for rep in range(6):
            rotas = [list(r) for r in base]
            # perturba: alguns movimentos relocate aleatorios viaveis (sem esvaziar rotas) para sair do ponto inicial
            clientes = [c for r in rotas for c in r]
            for _ in range(rng.randint(0, 40)):
                c = rng.choice(clientes)
                ri = next(i for i, r in enumerate(rotas) if c in r)
                rj = rng.randrange(len(rotas))
                if rj == ri or len(rotas[ri]) < 2:
                    continue
                carga = sum(inst.demands[x] for x in rotas[rj]) + inst.demands[c]
                if carga > inst.capacity:
                    continue
                rotas[ri].remove(c)
                rotas[rj].insert(rng.randint(0, len(rotas[rj])), c)
            c0 = custo(px, rotas)
            px.chamadas = 0
            m1, n1 = swap_star_best(px, [list(r) for r in rotas])
            calls_naive = px.chamadas
            px.chamadas = 0
            m2, n2 = swap_star_fast_best(px, [list(r) for r in rotas])
            calls_fast = px.chamadas
            assert m1 == m2, (nome, rep, "melhorou diverge", m1, m2)
            d1 = custo(px, n1) - c0 if m1 else 0.0
            d2 = custo(px, n2) - c0 if m2 else 0.0
            assert abs(d1 - d2) < 1e-7, (nome, rep, "melhor delta diverge", d1, d2)
            if m2:
                ok, msg = px.is_feasible([r for r in n2 if r])
                assert ok, (nome, rep, msg)
                assert sorted(c for r in n2 for c in r) == sorted(clientes)
                ok_improve += 1
            ratio_acc.append(calls_naive / max(calls_fast, 1))
            tot += 1
    print(f"{tot} solucoes testadas ({ok_improve} com melhora): mesmo melhor delta em todas; "
          f"razao de avaliacoes de distancia ingenuo/eficiente: media {sum(ratio_acc) / len(ratio_acc):.1f} "
          f"(min {min(ratio_acc):.1f}, max {max(ratio_acc):.1f})")
    print("TESTE DO SWAP* EFICIENTE: OK")


if __name__ == "__main__":
    main()
