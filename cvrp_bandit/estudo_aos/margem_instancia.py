#!/usr/bin/env python3
"""
margem_instancia.py
===================
Diagnostico EXPLORATORIO (pos hoc, nao pre-registrado; diario, Entrada 126) da margem que existiria para a SELECAO POR INSTANCIA
(escolher, para cada instancia, o melhor DESENHO entre os ja avaliados) sobre dados JA coletados dos Estudos 16 (ALNS no CVRPTW) e 17
(ILS do PyVRP). Nenhuma execucao nova. Tres estimadores, porque o "oraculo" ingenuo e viesado:

  (1) ORACULO OTIMISTA: para cada instancia, o menor gap medio entre os desenhos. Escolhe e avalia nos MESMOS dados: vies do vencedor
      (winner's curse); NAO e uma margem realizavel.
  (2) ORACULO COM VALIDACAO CRUZADA: divide as seeds ao acaso em duas metades A e B (200 divisoes); escolhe o melhor desenho de cada
      instancia em A e o avalia em B; compara com o melhor desenho UNICO escolhido em A (SBS), avaliado em B. Sem vies de selecao, mas
      com a escolha ruidosa (poucas seeds por metade): subestima a margem quando o ruido e grande.
  (3) COMPONENTES DE VARIANCIA: modelo y_{ias} = mu_a + alpha_i + u_{ia} + e_{ias}; sigma_u^2 (interacao instancia x desenho) por momentos
      das medias de celula duplamente centradas, descontado o ruido sigma_e^2/m; margem verdadeira (limite superior para QUALQUER seletor,
      inclusive com informacao perfeita) = E[min_a(mu_a + u_ia)] - min_a mu_a, por simulacao com u ~ N(0, sigma_u^2) independente entre
      desenhos (hipotese conservadora: desenhos correlacionados dao MENOS margem). IC95% por bootstrap das instancias.
  CALIBRACAO por simulacao: dados sinteticos com a mesma estrutura de ruido do Estudo 17 (nivel 1) e interacao conhecida tau; mostra que o
  oraculo com validacao cruzada so enxerga a margem quando tau >= 0,2 (margem verdadeira >= 0,175 pp), isto e, e um estimador CONSERVADOR.
  CONTROLE POSITIVO (inconclusivo, relatado como tal): escolher a temperatura inicial w0 do ALNS por instancia (SYN_TUNE_W, 3 seeds).

Saida: results/aos_study/margem_instancia.txt (stdout). Semente fixa. Uso: python estudo_aos/margem_instancia.py
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
rng = np.random.default_rng(1)


def cubo(d, arms):
    """dict instancia -> array [n_bracos, n_seeds] (so instancias com todos os bracos e seeds viaveis)"""
    d = d[d.viavel & d.braco.isin(arms)]
    out = {}
    for inst, g in d.groupby("instancia"):
        piv = g.pivot_table(index="braco", columns="seed", values="gap_pct")
        if piv.shape[0] == len(arms) and not piv.isna().any().any():
            out[inst] = piv.loc[arms].values
    return out


# ------------------------------------------------------------------------------------------ (1) e (2)
def margem_cv(cub, rotulo, nsplit=200):
    insts = sorted(cub)
    res = []
    for _ in range(nsplit):
        A, B = [], []
        for i in insts:
            ns = cub[i].shape[1]
            idx = rng.permutation(ns)
            h = ns // 2
            A.append(cub[i][:, idx[:h]].mean(axis=1))
            B.append(cub[i][:, idx[h:]].mean(axis=1))
        A, B = np.array(A), np.array(B)
        sbs = A.mean(axis=0).argmin()
        res.append((B[:, sbs].mean(), B[np.arange(len(insts)), A.argmin(axis=1)].mean(), B.min(axis=1).mean()))
    r = np.array(res)
    g = r[:, 0] - r[:, 1]
    print(f"{rotulo}: {len(insts)} instancias | SBS {r[:, 0].mean():.3f} | por instancia (cv) {r[:, 1].mean():.3f} | ganho cv {g.mean():+.3f} pp "
          f"[{np.percentile(g, 2.5):+.3f}, {np.percentile(g, 97.5):+.3f}] | oraculo otimista {np.mean(r[:, 0] - r[:, 2]):+.3f} pp")


def margem_cv_classe(cub, rotulo, nsplit=200):
    """escolha por CLASSE de Solomon (melhor desenho de cada classe em A; avalia em B)"""
    cl = lambda n: re.match(r"([A-Z]+)(\d)", n).group(1) + re.match(r"([A-Z]+)(\d)", n).group(2)  # noqa: E731
    insts = sorted(cub)
    classes = np.array([cl(i) for i in insts])
    res = []
    for _ in range(nsplit):
        A, B = [], []
        for i in insts:
            idx = rng.permutation(cub[i].shape[1])
            h = cub[i].shape[1] // 2
            A.append(cub[i][:, idx[:h]].mean(axis=1))
            B.append(cub[i][:, idx[h:]].mean(axis=1))
        A, B = np.array(A), np.array(B)
        sbs = A.mean(axis=0).argmin()
        escolha = np.zeros(len(insts), dtype=int)
        for c in set(classes):
            m = classes == c
            escolha[m] = A[m].mean(axis=0).argmin()
        res.append(B[:, sbs].mean() - B[np.arange(len(insts)), escolha].mean())
    print(f"{rotulo}: ganho cv da escolha por CLASSE sobre o melhor desenho unico {np.mean(res):+.3f} pp "
          f"[{np.percentile(res, 2.5):+.3f}, {np.percentile(res, 97.5):+.3f}]")


# ------------------------------------------------------------------------------------------ (3)
def estima(cub):
    insts = sorted(cub)
    I, K = len(insts), next(iter(cub.values())).shape[0]
    cel = np.array([cub[i].mean(axis=1) for i in insts])
    m = np.array([cub[i].shape[1] for i in insts])
    s2 = np.array([[cub[i][a].var(ddof=1) for a in range(K)] for i in insts])
    var_e = (s2.mean(axis=1) / m).mean()
    d = cel - cel.mean(axis=1, keepdims=True) - cel.mean(axis=0, keepdims=True) + cel.mean()
    c = (1 - 1 / K) * (1 - 1 / I)
    var_u = max(0.0, (d ** 2).mean() / c - var_e)
    return var_u, var_e, cel.mean(axis=0), K


def margem_verdadeira(var_u, mu, nsim=20000):
    u = rng.normal(0, np.sqrt(var_u), size=(nsim, len(mu)))
    return mu.min() - (mu[None, :] + u).min(axis=1).mean()


def margem_vc(cub, rotulo, nboot=300):
    var_u, var_e, mu, K = estima(cub)
    h = margem_verdadeira(var_u, mu)
    insts = sorted(cub)
    hs, sus = [], []
    for _ in range(nboot):
        amostra = rng.choice(len(insts), len(insts))
        sub = {f"{j}_{insts[k]}": cub[insts[k]] for j, k in enumerate(amostra)}
        v, _, mub, _ = estima(sub)
        sus.append(np.sqrt(v))
        hs.append(margem_verdadeira(v, mub, 3000))
    print(f"{rotulo}: K={K}, I={len(insts)} | dp do ruido da media de celula {np.sqrt(var_e):.3f} | dp da interacao {np.sqrt(var_u):.3f} "
          f"[{np.percentile(sus, 2.5):.3f}, {np.percentile(sus, 97.5):.3f}] | margem verdadeira (limite superior de qualquer seletor) {h:.3f} pp "
          f"[{np.percentile(hs, 2.5):.3f}, {np.percentile(hs, 97.5):.3f}]")


def main():
    e = pd.read_csv(RES / "etapa17_eval.csv")
    e["braco"] = e.braco.str.replace("s17_", "", regex=False)
    fixos = ["P3", "P4", "P5", "P6", "P7", "U", "static"]
    print("=== (1)+(2) oraculo otimista e oraculo com validacao cruzada (split das seeds, 200 divisoes) ===")
    for nv, lab in ((0.03, "E17 nivel 1 (0.03 s/cliente)"), (0.1, "E17 nivel 2 (0.10 s/cliente)")):
        d = e[e.nivel == nv]
        for nome, arms in (("P3-P7, U, estatica", fixos), ("todos os 14", sorted(d.braco.unique()))):
            cub = cubo(d, arms)
            margem_cv(cub, f"{lab} [{nome}]")
            margem_cv({i: v for i, v in cub.items() if not i.startswith("X-syn")}, "      CVRPLIB")
            margem_cv({i: v for i, v in cub.items() if i.startswith("X-syn")}, "      SYN_VAL")
    a = pd.read_csv(RES / "etapa16_eval_a.csv")
    a["braco"] = a.braco.str.replace("s16a_", "", regex=False)
    arms16 = sorted(a.braco.unique())
    margem_cv(cubo(a, arms16), "E16 ALNS-TW [todos os 10 bracos]")
    margem_cv(cubo(a, [x for x in arms16 if x != "eff"]), "E16 ALNS-TW [sem eficiencia]")
    c5 = cubo(a, ["U", "static", "fixrep", "RP", "pm"])
    margem_cv(c5, "E16 ALNS-TW [U, static, fixrep, RP, pm]")
    margem_cv_classe(c5, "E16 ALNS-TW [U, static, fixrep, RP, pm]")

    print("\n=== (3) componentes de variancia (limite superior sem vies do vencedor) ===")
    for nv, lab in ((0.03, "E17 nivel 1"), (0.1, "E17 nivel 2")):
        cub = cubo(e[e.nivel == nv], fixos)
        margem_vc(cub, f"{lab} [P3-P7, U, static] todas")
        margem_vc({i: v for i, v in cub.items() if not i.startswith("X-syn")}, f"{lab} CVRPLIB")
        margem_vc({i: v for i, v in cub.items() if i.startswith("X-syn")}, f"{lab} SYN_VAL")
    margem_vc(c5, "E16 ALNS-TW [U, static, fixrep, RP, pm]")
    margem_vc(cubo(a, ["U", "static", "fixrep"]), "E16 ALNS-TW [U, static, fixrep]")

    print("\n=== calibracao por simulacao (K=7, I=104, 5 seeds; ruido por celula como no E17 nivel 1) ===")
    cub1 = cubo(e[e.nivel == 0.03], fixos)
    _, var_e1, mu1, K1 = estima(cub1)
    sd_seed = np.sqrt(var_e1 * 7.5)
    for tau in (0.0, 0.05, 0.1, 0.2, 0.3):
        ganhos, verd = [], []
        for _ in range(60):
            I, m = 104, 5
            u = rng.normal(0, tau, size=(I, K1))
            y = mu1[None, :, None] + u[:, :, None] + rng.normal(0, sd_seed, size=(I, K1, m))
            A, B = y[:, :, :3].mean(axis=2), y[:, :, 3:].mean(axis=2)
            sbs = A.mean(axis=0).argmin()
            ganhos.append(B[:, sbs].mean() - B[np.arange(I), A.argmin(axis=1)].mean())
            verd.append(mu1.min() - (mu1[None, :] + u).min(axis=1).mean())
        print(f"   tau = {tau:.2f}: margem verdadeira {np.mean(verd):.3f} pp | ganho do oraculo cv {np.mean(ganhos):+.3f} pp (dp {np.std(ganhos):.3f})")

    print("\n=== controle positivo (INCONCLUSIVO): w0 do ALNS por instancia (SYN_TUNE_W, 3 seeds, 7 valores) ===")
    d = pd.concat([pd.read_csv(RES / "etapa16_piloto_a2.csv"), pd.read_csv(RES / "etapa16_piloto_a3.csv")])
    d = d[d.viavel]
    d["pol"] = d.braco.str.extract(r"p[bc]_([A-Za-z0-9]+)_w")[0]
    d["w0"] = d.braco.str.extract(r"_w([0-9.]+)$")[0].astype(float)
    for pol in ("U", "RP", "reg2"):
        g = d[d.pol == pol]
        ws = sorted(g.w0.unique())
        cub = {}
        for inst, h in g.groupby("instancia"):
            piv = h.pivot_table(index="w0", columns="seed", values="gap_pct").loc[ws]
            if not piv.isna().any().any():
                cub[inst] = piv.values
        insts = sorted(cub)
        res = []
        for B in range(3):
            A_idx = [i for i in range(3) if i != B]
            A = np.array([cub[i][:, A_idx].mean(axis=1) for i in insts])
            Bv = np.array([cub[i][:, B] for i in insts])
            sbs = A.mean(axis=0).argmin()
            res.append((Bv[:, sbs].mean(), Bv[np.arange(len(insts)), A.argmin(axis=1)].mean(), Bv.min(axis=1).mean()))
        r = np.array(res).mean(axis=0)
        print(f"   w0 por instancia ({pol}, {len(insts)} inst., {len(ws)} valores): w0 unico {r[0]:.3f} | por instancia (cv) {r[1]:.3f} | "
              f"ganho cv {r[0] - r[1]:+.3f} pp | oraculo otimista {r[0] - r[2]:+.3f} pp")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
