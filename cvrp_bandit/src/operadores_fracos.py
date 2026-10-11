"""
operadores_fracos.py
====================
Operadores ADICIONAIS para o Estudo 8 da Direcao A (controle positivo / "pool inflado"; ver diario).

Pergunta do estudo: quando o conjunto de operadores contem operadores ruins (a situacao em que um pesquisador
acrescenta operadores sem saber quais valem a pena), uma politica adaptativa protege a busca? Isto e tambem um
CONTROLE POSITIVO do protocolo: com um efeito plantado de direcao conhecida, o aparato experimental detecta o
ganho adaptativo (ou nao)?

Mesma interface de neighborhoods*.py: (nome, best_fn, random_fn), best_fn(instance, routes) -> (melhorou, novas_rotas).
O `instance` recebido e a InstanciaContadora (conta chamadas a dist em `instance.chamadas`).

  dud(frac)           operador INUTIL: gasta `frac * n^2` avaliacoes de distancia e nunca melhora (efeito plantado)
  restrito(base, r)   versao RESTRITA de um operador existente: aplica `base` a r rotas sorteadas (r=2) -- uma
                      aproximacao barata e fraca da vizinhanca completa (operador de "vizinhanca parcial")
"""
from neighborhoods_ext import _nao_usado_random


def dud(frac):
    nome = f"dud{frac:g}"

    def best_fn(instance, routes):
        instance.chamadas += int(round(frac * instance.n_customers ** 2))
        return False, routes

    return (nome, best_fn, _nao_usado_random)


def restrito(base, rng, r=2):
    nome_base, base_fn, _ = base
    nome = f"restr{r}({nome_base})"

    def best_fn(instance, routes):
        k = len(routes)
        if k < r:
            return False, routes
        sel = sorted(rng.sample(range(k), r))
        sub = [routes[i] for i in sel]
        melhorou, novo_sub = base_fn(instance, sub)
        if not melhorou:
            return False, routes
        resto = [rt for i, rt in enumerate(routes) if i not in set(sel)]
        return True, resto + [rt for rt in novo_sub if rt]

    return (nome, best_fn, _nao_usado_random)
