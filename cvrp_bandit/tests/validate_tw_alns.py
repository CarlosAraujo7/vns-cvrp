#!/usr/bin/env python3
"""
validate_tw_alns.py
===================
Valida src/tw_alns.py (ALNS do CVRPTW, Estudo 16) contra implementacoes INDEPENDENTES por forca bruta que usam so a simulacao exata
de rota (TWInst.viavel_rota / custo_rota), nunca os vetores de tempo do estado:
  (A) melhor posicao de insercao: para cada cliente removido e cada rota, `_opcoes_tw` (viabilidade O(1) por posicao com inicio de servico
      e inicio mais tardio) = enumeracao de todas as posicoes com `viavel_rota` (delta, posicao do primeiro minimo, ausencia de opcao), e
      a contagem de avaliacoes de distancia (len(nos) - 1 por rota examinada, so se a capacidade permite);
  (B) os tres reparos: o resultado exato (rotas e posicoes) de guloso, regret-2 e barato = reparo lento reescrito com a enumeracao;
  (C) contabilidade: soma das arestas do estado = custo recomputado; viabilidade `ok[i]` = simulacao; cobertura dos clientes; se a parcial
      era viavel e o reparo conclui, a solucao e viavel pela simulacao;
  (D) limite inferior: o ALNS completo devolve solucao viavel e custo >= BKS (otimo provado) nas instancias de Solomon.
Estados: subinstancias aleatorias de Solomon (8 a 24 clientes), solucoes de construcao e apos iteracoes de ALNS, apos cada uma das
quatro destruicoes, e copias com JANELAS e CAPACIDADE APERTADAS (a igualdade passa a importar).
Mutacoes (copias do modulo com uma falha injetada) devem ser detectadas pelo mesmo teste.

Uso (WSL, venv): python tests/validate_tw_alns.py
"""
import importlib.util
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))
from tw_inst import TWInst, ler_solomon, construcao_inicial  # noqa: E402
from validate_tw_neighborhoods import sub_instancia, aperta_janelas  # noqa: E402
import tw_alns as real  # noqa: E402

INF = float("inf")


# ----------------------------------------------------------------------------------------- referencia independente
def pos_forca_bruta(inst, r, c):
    """(delta, posicao do primeiro minimo) sobre TODAS as posicoes viaveis (capacidade e janelas, por simulacao), ou None."""
    base = inst.custo_rota(r)
    melhor, bp = None, None
    for p in range(len(r) + 1):
        nova = r[:p] + [c] + r[p:]
        if not inst.viavel_rota(nova):
            continue
        d = inst.custo_rota(nova) - base
        if melhor is None or d < melhor:
            melhor, bp = d, p
    return (melhor, bp) if bp is not None else None


def reparo_lento(inst, rotas, removidos, k, max_rotas):
    rotas = [list(r) for r in rotas]
    pend = list(removidos)
    novo = {c: 2 * int(inst.D[0][c]) for c in pend}
    while pend:
        op = {}
        for c in pend:
            op[c] = {}
            for i, r in enumerate(rotas):
                res = pos_forca_bruta(inst, r, c)
                if res:
                    op[c][i] = res
        pode_abrir = max_rotas is None or len(rotas) < max_rotas
        melhor = None
        for c in pend:
            t = sorted([(v[0], i) for i, v in op[c].items()] + ([(novo[c], -1)] if pode_abrir else []))
            if not t:
                return None
            if k == 1:
                chave = (t[0][0], 0)
            elif len(t) == 1:
                chave = (-INF, t[0][0])
            else:
                chave = (-(t[1][0] - t[0][0]), t[0][0])
            if melhor is None or chave < melhor[0]:
                melhor = (chave, c, t[0])
        _, c, esc = melhor
        pend.remove(c)
        if esc[1] == -1:
            rotas.append([c])
        else:
            i, pos = esc[1], op[c][esc[1]][1]
            rotas[i] = rotas[i][:pos] + [c] + rotas[i][pos:]
    return rotas


def reparo_barato_lento(inst, rotas, removidos, rng, max_rotas):
    rotas = [list(r) for r in rotas]
    ordem = list(removidos)
    rng.shuffle(ordem)
    for c in ordem:
        op = {}
        for i, r in enumerate(rotas):
            res = pos_forca_bruta(inst, r, c)
            if res:
                op[i] = res
        mi = min(op, key=lambda i: (op[i][0], i)) if op else None
        custo_novo = 2 * int(inst.D[0][c])
        pode_abrir = max_rotas is None or len(rotas) < max_rotas
        if mi is None and not pode_abrir:
            return None
        if mi is None or (pode_abrir and custo_novo < op[mi][0]):
            rotas.append([c])
        else:
            rotas[mi] = rotas[mi][:op[mi][1]] + [c] + rotas[mi][op[mi][1]:]
    return rotas


# ----------------------------------------------------------------------------------------- estados
def estados(mod, base, rng, n_est):
    saida = []
    for _ in range(n_est):
        n = rng.randint(8, 24)
        sub = sub_instancia(base, rng, n)
        assert all(sub.viavel_rota([c]) for c in range(1, sub.N)), "cliente inviavel sozinho"
        inst = mod.TWAlnsInst.desde(sub)
        x = construcao_inicial(sub)
        saida.append((inst, x))
        y = x
        for _k in range(rng.randint(2, 8)):                      # iteracoes de ALNS para diversificar
            est = mod.EstadoTW(inst, y, 1.0)
            rem = mod.DESTRUIR_TW[rng.randrange(4)][1](est, rng.randint(2, max(2, n // 3)), rng)
            if mod.REPARAR_TW[rng.randrange(3)][1](est, rem, rng, sub.nveic) and sub.is_feasible(est.rotas)[0]:
                y = [list(r) for r in est.rotas]
        saida.append((inst, y))
        saida.append((mod.TWAlnsInst.desde(aperta_janelas(sub, y, rng)), y))
    return saida


def validar(mod, base, rng, n_est=40, verbose=True):
    f = {"opcoes": 0, "contagem": 0, "ok": 0, "reparo": 0, "custo": 0, "cobertura": 0, "viavel": 0}
    v = {k: 0 for k in f}
    for inst, y in estados(mod, base, rng, n_est):
        dmax = max(max(r) for r in inst._D) or 1
        for jd in range(4):
            est = mod.EstadoTW(inst, y, dmax)
            q = rng.randint(2, max(2, inst.n_customers // 3))
            rem = (mod.DESTRUIR_TW[jd][1](est, q, rng) if jd != 3 else mod.DESTRUIR_TW[jd][1](est, q, rng, qmax=q))
            if not rem:
                continue
            # (C) viabilidade ok[i] = simulacao de tempo (sem capacidade); custo das arestas
            semcap = TWInst(inst.nome, inst.coords, inst.dem, inst.e, inst.l, inst.s, 10 ** 9, inst.nveic)
            tempo_ok = [semcap.viavel_rota(r) for r in est.rotas]
            v["ok"] += 1
            if tempo_ok != list(est.ok):
                f["ok"] += 1
            if sum(map(sum, est.arest)) != inst.solution_cost(est.rotas):
                f["custo"] += 1
            v["custo"] += 1
            todas_ok = all(est.ok)
            # (A) opcoes de insercao
            for c in rem:
                antes = inst.chamadas
                got = mod._opcoes_tw(est, c)
                esperado, cont = {}, 0
                for i, r in enumerate(est.rotas):
                    if not est.ok[i] or est.cargas[i] + est.dem[c] > est.Q:
                        continue
                    cont += len(est.nos[i]) - 1
                    res = pos_forca_bruta(inst, r, c)
                    if res:
                        esperado[i] = res
                v["opcoes"] += 1
                if got != esperado:
                    f["opcoes"] += 1
                v["contagem"] += 1
                if inst.chamadas - antes != cont:
                    f["contagem"] += 1
            # (B) reparos: resultado exato; (C) cobertura e viabilidade
            parcial = [list(r) for r in est.rotas]
            for jr in range(3):
                seed = rng.randrange(10 ** 6)
                e2 = mod.EstadoTW(inst, parcial, dmax)
                r2 = random.Random(seed)
                ok_real = mod.REPARAR_TW[jr][1](e2, list(rem), r2, inst.nveic)
                if jr < 2:
                    lento = reparo_lento(inst, parcial, rem, jr + 1, inst.nveic)
                else:
                    lento = reparo_barato_lento(inst, parcial, rem, random.Random(seed), inst.nveic)
                v["reparo"] += 1
                if (lento is None) != (not ok_real) or (lento is not None and [list(r) for r in e2.rotas] != lento):
                    f["reparo"] += 1
                if ok_real:
                    v["cobertura"] += 1
                    if sorted(c for r in e2.rotas for c in r) != list(range(1, inst.N)):
                        f["cobertura"] += 1
                    if sum(map(sum, e2.arest)) != inst.solution_cost(e2.rotas):
                        f["custo"] += 1
                    v["custo"] += 1
                    if todas_ok:
                        v["viavel"] += 1
                        if not inst.is_feasible(e2.rotas)[0]:
                            f["viavel"] += 1
    total = sum(f.values())
    if verbose:
        for k in f:
            print(f"  {k:10s} {f[k]:4d} falhas em {v[k]} verificacoes")
    return total, sum(v.values()), f


def alns_completo(mod):
    """(D) ALNS inteiro, orcamento curto, em instancias reais de Solomon: viavel e custo >= BKS."""
    import csv
    from alns_loop import rodar_alns, construir_politica_alns
    refs = {r["instancia"]: int(float(r["ref"])) for r in csv.DictReader(open(RAIZ / "instances" / "SOLOMON" / "ref.csv"))}
    n_inv = 0
    for nome in ("C101", "R101", "RC101", "R205", "RC208"):
        base = ler_solomon(RAIZ / "instances" / "SOLOMON" / f"{nome}.txt", ref=refs[nome])
        inst = mod.TWAlnsInst.desde(base)
        x0 = construcao_inicial(base)
        rng = random.Random(7)
        pd_, md = construir_politica_alns("U", 4, random.Random(1))
        pr_, mr = construir_politica_alns("U", 3, random.Random(2))
        best, info = rodar_alns(inst, [list(r) for r in x0], rng, 60 * 100 * 100, pd_, md, pr_, mr, lambda: inst.chamadas,
                                w0=0.02, max_rotas=inst.nveic, destruir=mod.DESTRUIR_TW, reparar=mod.REPARAR_TW,
                                estado_cls=mod.EstadoTW)
        ok, msg = base.is_feasible(best)
        custo = base.solution_cost(best)
        print(f"  {nome}: custo {custo / 10:.1f} (BKS {refs[nome] / 10:.1f}, inicial {base.solution_cost(x0) / 10:.1f}) viavel={ok} "
              f"iteracoes={info['iteracoes']} inviaveis={info['inviaveis']}")
        assert ok, (nome, msg)
        assert custo >= refs[nome], f"custo abaixo do otimo provado em {nome}: bug de custo ou viabilidade"
        assert custo <= base.solution_cost(x0)
        n_inv += info["inviaveis"]
    return n_inv


def modulo_mutado(nome, subs):
    src = (RAIZ / "src" / "tw_alns.py").read_text(encoding="utf-8")
    for old, new in subs:
        assert src.count(old) == 1, (nome, old, src.count(old))
        src = src.replace(old, new, 1)
    dest = RAIZ / "tests" / f"_mut_twalns_{nome}.py"
    dest.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"_mut_twalns_{nome}", dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, dest


MUTACOES = {
    "sem_L": [("        if t > lc or t + sc + Dc[v] > L[p]:", "        if t > lc:")],
    "sem_lc": [("        if t > lc or t + sc + Dc[v] > L[p]:", "        if t + sc + Dc[v] > L[p]:")],
    "estrita_L": [("t + sc + Dc[v] > L[p]:", "t + sc + Dc[v] >= L[p]:")],
    "sem_espera": [("        if t < ec:\n            t = ec\n", "")],
    "contagem": [("est.inst.chamadas += len(nos) - 1", "est.inst.chamadas += len(nos)")],
    "L_sem_servico": [("x = L[k + 1] - s[v] - D[v][w]", "x = L[k + 1] - D[v][w]")],
    "capacidade": [("if est.ok[i] and est.cargas[i] + est.dem[c] <= est.Q:\n            r = melhor_posicao_tw(est, i, c)\n            if r is not None:\n                out[i] = r",
                    "if est.ok[i] and est.cargas[i] + est.dem[c] <= est.Q - 1:\n            r = melhor_posicao_tw(est, i, c)\n            if r is not None:\n                out[i] = r")],
    "regret_sinal": [("chave = (-(custos[1][0] - custos[0][0]), custos[0][0])", "chave = ((custos[1][0] - custos[0][0]), custos[0][0])")],
    "sem_tempo_delta": [("delta = duc + Dc[v] - arest[p - 1]", "delta = duc + Dc[v] - arest[p - 1] + (1 if p == 1 else 0)")],
}


def main():
    base = ler_solomon(RAIZ / "instances" / "SOLOMON" / "RC101.txt")
    print("Modulo real:")
    total, verific, f = validar(real, base, random.Random(2026), n_est=40)
    print(f"RESULTADO: {total} falhas em {verific} verificacoes")
    assert total == 0
    print("ALNS completo (viabilidade e limite inferior):")
    alns_completo(real)
    for nome, subs in MUTACOES.items():
        mod, dest = modulo_mutado(nome, subs)
        try:
            t, v, _ = validar(mod, base, random.Random(7), n_est=14, verbose=False)
        finally:
            dest.unlink(missing_ok=True)
        print(f"  mutacao {nome}: {t} falhas detectadas em {v} verificacoes")
        assert t > 0, f"mutacao {nome} NAO foi detectada"
    print("OK")


if __name__ == "__main__":
    main()
