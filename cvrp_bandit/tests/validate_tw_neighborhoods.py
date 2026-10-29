#!/usr/bin/env python3
"""
validate_tw_neighborhoods.py
============================
Valida os operadores de src/tw_neighborhoods.py (numba, janelas de tempo por concatenacao de segmentos) contra uma ENUMERACAO
INDEPENDENTE por forca bruta: para cada candidato da vizinhanca (na mesma ordem de varredura), as rotas novas sao CONSTRUIDAS
explicitamente, a viabilidade (capacidade e janelas de tempo) e verificada por SIMULACAO da rota (inst.viavel_rota) e o delta e
recomputado do zero (inst.custo_rota). Confere, por operador e estado: (a) se ha movimento de melhora; (b) as rotas resultantes,
exatamente (inclui o desempate: primeiro da varredura); (c) o custo do resultado = custo antigo + delta recomputado; (d) o numero
de avaliacoes de distancia somado a inst.chamadas = candidatos que passam na capacidade x arcos novos do operador.
Estados: subconjuntos aleatorios de instancias de Solomon (8 a 24 clientes), solucao inicial, estados apos movimentos aleatorios
viaveis (shake) e apos movimentos de melhora.

Mutacoes: copias do modulo com uma falha injetada (viabilidade temporal estrita, contagem, arco do delta, teste de capacidade)
devem ser detectadas pelo mesmo teste.

Uso: python tests/validate_tw_neighborhoods.py
"""
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from tw_inst import TWInst, ler_solomon, construcao_inicial  # noqa: E402
import tw_neighborhoods as nb_real  # noqa: E402


def sub_instancia(base, rng, n):
    ids = sorted(rng.sample(range(1, base.N), n))
    sel = [0] + ids
    return TWInst(f"{base.nome}-sub{n}", base.coords[sel], base.dem[sel], base.e[sel], base.l[sel], base.s[sel], base.Q, 25)


# ----------------------------------------------------------------------------------------- enumeracao independente
def carga(inst, r):
    return int(sum(inst.dem[c] for c in r))


def cust(inst, rs):
    return sum(inst.custo_rota(r) for r in rs)


def ref_melhor(inst, rs, candidatos):
    """candidatos: gerador de (indices_das_rotas_alteradas, novas_rotas_para_esses_indices, n_arcos) na ordem de varredura e
    ja filtrados pela capacidade. Devolve (delta, rotas_resultantes ou None, contagem)."""
    melhor, res, cnt = 0, None, 0
    for idx, novas, arcos in candidatos:
        cnt += arcos
        if not all(inst.viavel_rota(r) for r in novas):
            continue
        delta = sum(inst.custo_rota(r) for r in novas) - sum(inst.custo_rota(rs[k]) for k in idx)
        if delta < melhor:
            nrs = [list(r) for r in rs]
            for k, r in zip(idx, novas):
                nrs[k] = r
            melhor, res = delta, [r for r in nrs if r]
    return melhor, res, cnt


def cand_reloc1(inst, rs):
    m = len(rs)
    for a in range(m):
        for i in range(len(rs[a])):
            u = rs[a][i]
            A = rs[a][:i] + rs[a][i + 1:]
            for b in range(m):
                if b == a or carga(inst, rs[b]) + inst.dem[u] > inst.Q:
                    continue
                for j in range(len(rs[b]) + 1):
                    yield (a, b), (A, rs[b][:j] + [u] + rs[b][j:]), 3


def cand_reloc2(inst, rs):
    m = len(rs)
    for a in range(m):
        for i in range(len(rs[a]) - 1):
            u1, u2 = rs[a][i], rs[a][i + 1]
            A = rs[a][:i] + rs[a][i + 2:]
            for b in range(m):
                if b == a or carga(inst, rs[b]) + inst.dem[u1] + inst.dem[u2] > inst.Q:
                    continue
                for j in range(len(rs[b]) + 1):
                    for o in range(2):
                        seg = [u1, u2] if o == 0 else [u2, u1]
                        yield (a, b), (A, rs[b][:j] + seg + rs[b][j:]), 3 if o == 0 else 3   # 6 por (a,i,b,j): ver abaixo


def cand_swap11(inst, rs):
    m = len(rs)
    for a in range(m):
        for b in range(a + 1, m):
            for i in range(len(rs[a])):
                for j in range(len(rs[b])):
                    u, v = rs[a][i], rs[b][j]
                    if carga(inst, rs[a]) - inst.dem[u] + inst.dem[v] > inst.Q or carga(inst, rs[b]) - inst.dem[v] + inst.dem[u] > inst.Q:
                        continue
                    A = rs[a][:i] + [v] + rs[a][i + 1:]
                    B = rs[b][:j] + [u] + rs[b][j + 1:]
                    yield (a, b), (A, B), 4


def cand_twoopt(inst, rs):
    m = len(rs)
    for a in range(m):
        for b in range(a + 1, m):
            for i in range(len(rs[a]) + 1):
                for j in range(len(rs[b]) + 1):
                    A = rs[a][:i] + rs[b][j:]
                    B = rs[b][:j] + rs[a][i:]
                    if carga(inst, A) > inst.Q or carga(inst, B) > inst.Q:
                        continue
                    yield (a, b), (A, B), 2


def cand_move1(inst, rs):
    for a in range(len(rs)):
        La = len(rs[a])
        for i in range(1, La + 1):
            u = rs[a][i - 1]
            for j in range(0, La + 1):
                if j == i - 1 or j == i:
                    continue
                r = list(rs[a])
                if j < i - 1:
                    del r[i - 1]
                    r.insert(j, u)
                else:
                    r.insert(j, u)
                    del r[i - 1]
                yield (a,), (r,), 3


def cand_or23(inst, rs):
    for a in range(len(rs)):
        La = len(rs[a])
        for k in (2, 3):
            for i in range(1, La - k + 2):
                seg0 = rs[a][i - 1:i - 1 + k]
                for j in range(0, La + 1):
                    if i - 1 <= j <= i + k - 1:
                        continue
                    for o in range(2):
                        seg = seg0 if o == 0 else seg0[::-1]
                        r = list(rs[a])
                        if j < i - 1:
                            del r[i - 1:i - 1 + k]
                            r[j:j] = seg
                        else:
                            r[j:j] = seg
                            del r[i - 1:i - 1 + k]
                        yield (a,), (r,), 3


def cand_swapintra(inst, rs):
    for a in range(len(rs)):
        La = len(rs[a])
        for i in range(1, La):
            for j in range(i + 1, La + 1):
                r = list(rs[a])
                r[i - 1], r[j - 1] = r[j - 1], r[i - 1]
                yield (a,), (r,), 4


def cand_cross(maxlen, minmax):
    def g(inst, rs):
        m = len(rs)
        for a in range(m):
            for b in range(a + 1, m):
                for i in range(1, len(rs[a]) + 1):
                    for la in range(1, maxlen + 1):
                        if i + la - 1 > len(rs[a]):
                            break
                        for j in range(1, len(rs[b]) + 1):
                            for lb in range(1, maxlen + 1):
                                if j + lb - 1 > len(rs[b]):
                                    break
                                if max(la, lb) < minmax:
                                    continue
                                sa, sb = rs[a][i - 1:i - 1 + la], rs[b][j - 1:j - 1 + lb]
                                A = rs[a][:i - 1] + sb + rs[a][i - 1 + la:]
                                B = rs[b][:j - 1] + sa + rs[b][j - 1 + lb:]
                                if carga(inst, A) > inst.Q or carga(inst, B) > inst.Q:
                                    continue
                                yield (a, b), (A, B), 4
    return g


# Relocate(2,0): o kernel soma 6 por (a,i,b,j) que passa na capacidade (as duas orientacoes juntas); o gerador acima emite 2
# candidatos com 3 cada, de mesma soma. Or-opt(2-3): 6 por (a,k,i,j) = 3 por orientacao (idem).
ENUM = {"Move(1)": cand_move1, "Swap(1,1)": cand_swap11, "Relocate(1,0)": cand_reloc1, "Relocate(2,0)": cand_reloc2,
        "2-opt*": cand_twoopt, "Or-opt(2-3)": cand_or23, "Swap-intra": cand_swapintra,
        "CROSS(<=3)": cand_cross(3, 2), "CROSS(4-5)": cand_cross(5, 4)}


# ----------------------------------------------------------------------------------------- estados e comparacao
def aperta_janelas(inst, rotas, rng, frac=0.5):
    """Copia da instancia em que, para uma fracao dos clientes, a janela final vira o instante de inicio do servico na solucao dada
    (a solucao continua viavel, mas qualquer atraso, ainda que de 0,1, a torna inviavel; a igualdade e viavel). A capacidade vira a maior
    carga de rota da solucao."""
    l = inst.l.copy()
    for r in rotas:
        t, prev = 0, 0
        for c in r:
            t = max(t + inst.D[prev, c], inst.e[c])
            if rng.random() < frac:
                l[c] = t
            t += inst.s[c]
            prev = c
    Q = max(carga(inst, r) for r in rotas)          # a capacidade tambem vira a maior carga: a igualdade passa a importar
    return TWInst(inst.nome + "-ap", inst.coords, inst.dem, inst.e, l, inst.s, Q, inst.nveic)


def estados(base, rng, n_est):
    saida = []
    for _ in range(n_est):
        n = rng.randint(8, 24)
        inst = sub_instancia(base, rng, n)
        x = construcao_inicial(inst)
        saida.append((inst, x))
        y = x
        for _k in range(rng.randint(2, 10)):
            nome, fn, rf = nb_real.P5[rng.randrange(5)]
            z = rf(inst, y, rng)
            if z is not None:
                y = z
        saida.append((inst, y))
        w = y
        for _k in range(rng.randint(1, 3)):
            ok, w2 = nb_real.P9[rng.randrange(9)][1](inst, w)
            if ok:
                w = w2
        saida.append((inst, w))
        saida.append((aperta_janelas(inst, w, rng), w))      # janelas finais iguais ao instante de servico: a igualdade passa a importar
    return saida


def validar(nb, base, rng, n_est=60, verbose=True):
    falhas = {nome: 0 for nome in ENUM}
    verific = 0
    melhorou = {nome: 0 for nome in ENUM}
    for inst, x in estados(base, rng, n_est):
        assert inst.is_feasible(x)[0]
        for nome, fn, _rf in nb.P9:
            inst.chamadas = 0
            ok, y = fn(inst, x)
            cont = inst.chamadas
            delta_e, res_e, cont_e = ref_melhor(inst, x, ENUM[nome](inst, x))
            verific += 1
            bad = (cont != cont_e) or (ok != (res_e is not None))
            if ok and res_e is not None:
                bad = bad or ([list(r) for r in y] != res_e)
                bad = bad or (inst.solution_cost(y) != inst.solution_cost(x) + delta_e)
                bad = bad or (not inst.is_feasible(y)[0])
                melhorou[nome] += 1
            if bad:
                falhas[nome] += 1
    total = sum(falhas.values())
    if verbose:
        print(f"  {verific} verificacoes; falhas por operador: {falhas}; estados com melhora: {melhorou}")
    return total, verific, melhorou


def modulo_mutado(nome, subs):
    src = (RAIZ / "src" / "tw_neighborhoods.py").read_text(encoding="utf-8")
    for old, new in subs:
        assert src.count(old) >= 1, (nome, old)
        src = src.replace(old, new, 1)
    dest = RAIZ / "tests" / f"_mut_tw_{nome}.py"
    dest.write_text(src.replace("cache=True", "cache=False"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"_mut_tw_{nome}", dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, dest


MUTACOES = {
    "tw_estrita": [("ok = A[1] + dd <= B[2]", "ok = A[1] + dd < B[2]")],
    "cont_reloc1": [("                    cnt += 3\n                    if not okA:", "                    cnt += 2\n                    if not okA:")],
    "arco_swap11": [("delta = Dm[pa, v] + Dm[v, sa] - Dm[pa, u] - Dm[u, sa] + Dm[pb, u] + Dm[u, sb] - Dm[pb, v] - Dm[v, sb]\n                    if delta >= best:\n                        continue\n                    S1, ok1 = cat(pre(PD, PE, PL, R, a, i - 1), no(v, e, l, s), Dm)",
                    "delta = Dm[pa, v] + Dm[v, sa] - Dm[pa, u] - Dm[u, sa] + Dm[pb, u] + Dm[u, sb] - Dm[pb, v] - Dm[sa, sb]\n                    if delta >= best:\n                        continue\n                    S1, ok1 = cat(pre(PD, PE, PL, R, a, i - 1), no(v, e, l, s), Dm)")],
    "cap_twoopt": [("if nA > Q or nB > Q:", "if nA >= Q or nB > Q:")],
    "cont_cross": [("                            cnt += 4\n                            pb = R[b, j - 1]\n                            sb = R[b, j + lb]", "                            cnt += 5\n                            pb = R[b, j - 1]\n                            sb = R[b, j + lb]")],
}


def main():
    base = ler_solomon(RAIZ / "instances" / "SOLOMON" / "RC101.txt")
    rng = random.Random(2026)
    print("Modulo real:")
    total, verific, melhorou = validar(nb_real, base, rng, n_est=80)
    print(f"RESULTADO: {total} falhas em {verific} verificacoes")
    assert total == 0
    assert all(v > 0 for v in melhorou.values()), "algum operador nunca encontrou melhora: estados pouco informativos"
    for nome, subs in MUTACOES.items():
        mod, dest = modulo_mutado(nome, subs)
        try:
            t, v, _ = validar(mod, base, random.Random(7), n_est=25, verbose=False)
        finally:
            dest.unlink(missing_ok=True)
        print(f"  mutacao {nome}: {t} falhas detectadas em {v} verificacoes")
        assert t > 0, f"mutacao {nome} NAO foi detectada"
    print("OK")


if __name__ == "__main__":
    main()
