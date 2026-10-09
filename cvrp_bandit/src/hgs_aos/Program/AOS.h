// AOS.h -- adaptive selection of the move type inside the HGS-CVRP local search (Study 7 of the AOS audit;
// research log entry 97). NOT part of the original HGS-CVRP (T. Vidal, MIT license, see ../LICENSE).
//
// For every pair of nodes (U,V) the original local search tries the nine route-improvement moves in a fixed order
// (relocate x3, swap x3, 2-opt, 2-opt* x2) and applies the first improving one. Here a policy chooses the order in
// which the APPLICABLE moves of the pair are tried (policy.select among the moves not yet tried for this pair) and is
// told whether each attempt found an improving move (binary reward). The policy state persists over the whole run.
//
// Statistics are updated after EVERY attempt; the indices used for selection (UCB scores, Thompson draws, softmax
// probabilities) are recomputed every R attempts (R = 16 during the first 4096 attempts, then 256): with 1e7-1e8 attempts
// per run this "batched refresh" keeps the policy overhead small and is a deliberate, documented approximation.
// aosMode: 0 = original code path (policy not used), 1 uniform random order, 2 UCB1, 3 Thompson sampling,
//          4 probability matching, 5 HG-UCB-lite (no ADWIN drift detector, binary reward).
#ifndef AOS_H
#define AOS_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

class AosPolicy
{
public:
	static const int K = 9;
	unsigned long long state;
	long long attempts[K];
	long long successes[K];
	long long total;
	long long nextRefresh;

	AosPolicy(unsigned seed) : total(0), nextRefresh(16)
	{
		state = 0x9E3779B97F4A7C15ULL ^ ((unsigned long long)seed * 0xBF58476D1CE4E5B9ULL);
		for (int j = 0; j < K; j++) { attempts[j] = 0; successes[j] = 0; }
		for (int i = 0; i < 8; i++) next();
	}
	virtual ~AosPolicy() {}

	// chooses one of the na move ids stored in avail[0..na-1]
	virtual int select(const int * avail, int na) = 0;

	// reports the outcome of an attempt of move j (ok = an improving move was found and applied)
	virtual void update(int j, bool ok)
	{
		attempts[j]++;
		if (ok) successes[j]++;
		total++;
		if (total >= nextRefresh)
		{
			refresh();
			nextRefresh = total + (total < 4096 ? 16 : 256);
		}
	}

	virtual void refresh() {}

	void writeStats(const char * path) const
	{
		FILE * f = fopen(path, "w");
		if (!f) return;
		fprintf(f, "move,attempts,successes\n");
		for (int j = 0; j < K; j++) fprintf(f, "%d,%lld,%lld\n", j + 1, attempts[j], successes[j]);
		fclose(f);
	}

protected:
	unsigned long long next()               // xorshift64*
	{
		state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
		return state * 2685821657736338717ULL;
	}
	int uniformIndex(int n) { return (int)((next() >> 33) % (unsigned long long)n); }
	double unif01() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
	double normal01()                       // Box-Muller
	{
		double u1 = std::max(unif01(), 1e-300), u2 = unif01();
		return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
	}
	double gamma(double a)                  // Marsaglia-Tsang (a >= 1)
	{
		double d = a - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
		for (;;)
		{
			double x = normal01(), v = 1.0 + c * x;
			if (v <= 0) continue;
			v = v * v * v;
			double u = unif01();
			if (std::log(std::max(u, 1e-300)) < 0.5 * x * x + d - d * v + d * std::log(v)) return d * v;
		}
	}
	// a move id chosen at random among the applicable moves never attempted, or -1
	int untried(const int * avail, int na)
	{
		int cand[K], nc = 0;
		for (int i = 0; i < na; i++) if (attempts[avail[i]] == 0) cand[nc++] = avail[i];
		return nc ? cand[uniformIndex(nc)] : -1;
	}
	double rate(int j) const { return attempts[j] ? (double)successes[j] / (double)attempts[j] : 0.0; }
};

class UniformPolicy : public AosPolicy
{
public:
	UniformPolicy(unsigned seed) : AosPolicy(seed) {}
	int select(const int * avail, int na) { return avail[uniformIndex(na)]; }
};

// UCB1 on the success rate normalized by the largest rate (scale-free), exploration term C sqrt(2 ln s / n_j)
class Ucb1Policy : public AosPolicy
{
public:
	double C, score[K];
	Ucb1Policy(unsigned seed, double c = 1.0) : AosPolicy(seed), C(c) { for (int j = 0; j < K; j++) score[j] = 0.0; }
	void refresh()
	{
		double rmax = 0.0, ln = std::log((double)total + 1.0);
		for (int j = 0; j < K; j++) rmax = std::max(rmax, rate(j));
		for (int j = 0; j < K; j++)
			score[j] = attempts[j] ? (rmax > 0 ? rate(j) / rmax : 0.0) + C * std::sqrt(2.0 * ln / (double)attempts[j]) : 1e300;
	}
	int select(const int * avail, int na)
	{
		int u = untried(avail, na);
		if (u >= 0) return u;
		int bestId = avail[0];
		for (int i = 1; i < na; i++) if (score[avail[i]] > score[bestId]) bestId = avail[i];
		return bestId;
	}
};

// Thompson sampling, Beta(1+s,1+f) posterior (normal approximation when n >= 100); one draw per refresh and move
class ThompsonPolicy : public AosPolicy
{
public:
	double theta[K];
	ThompsonPolicy(unsigned seed) : AosPolicy(seed) { for (int j = 0; j < K; j++) theta[j] = 0.5; }
	void refresh()
	{
		for (int j = 0; j < K; j++)
		{
			long long s = successes[j], f = attempts[j] - successes[j];
			if (s + f < 100)
			{
				double x = gamma((double)(1 + s) < 1 ? 1.0 : (double)(1 + s)), y = gamma((double)(1 + f));
				theta[j] = x / (x + y);
			}
			else
			{
				double n = (double)(s + f), mu = ((double)s + 1.0) / (n + 2.0);
				theta[j] = mu + std::sqrt(mu * (1.0 - mu) / (n + 3.0)) * normal01();
			}
		}
	}
	int select(const int * avail, int na)
	{
		int bestId = avail[0];
		for (int i = 1; i < na; i++) if (theta[avail[i]] > theta[bestId]) bestId = avail[i];
		return bestId;
	}
};

// Probability matching: Q_j <- Q_j + gamma (r - Q_j); p_j = p_min + (1 - na p_min) Q_j / sum Q (uniform if sum Q = 0)
class ProbMatchingPolicy : public AosPolicy
{
public:
	double gamma_, pmin;
	double Q[K];
	ProbMatchingPolicy(unsigned seed, double g = 0.1, double p = 0.05) : AosPolicy(seed), gamma_(g), pmin(p)
	{
		for (int j = 0; j < K; j++) Q[j] = 0.0;
	}
	int select(const int * avail, int na)
	{
		double sum = 0.0;
		for (int i = 0; i < na; i++) sum += Q[avail[i]];
		if (sum <= 0.0) return avail[uniformIndex(na)];
		double r = unif01(), acc = 0.0;
		for (int i = 0; i < na; i++)
		{
			acc += pmin + (1.0 - na * pmin) * Q[avail[i]] / sum;
			if (r < acc) return avail[i];
		}
		return avail[na - 1];
	}
	void update(int j, bool ok)
	{
		AosPolicy::update(j, ok);
		Q[j] += gamma_ * ((ok ? 1.0 : 0.0) - Q[j]);
	}
};

// HG-UCB-lite: cold start, epsilon-greedy over the pool {q >= b_t} (all applicable moves if empty), otherwise UCB over the
// softmax preferences; gradient-bandit update with the mean reward as baseline; no ADWIN drift detector.
class HgUcbPolicy : public AosPolicy
{
public:
	double alpha, C, eps, bt, H[K], q[K], score[K], rbar;
	HgUcbPolicy(unsigned seed, double a = 0.5, double c = 2.0, double e = 0.1, double b = 0.2)
		: AosPolicy(seed), alpha(a), C(c), eps(e), bt(b), rbar(0.0)
	{
		for (int j = 0; j < K; j++) { H[j] = 0.0; q[j] = 1.0 / K; score[j] = 1e300; }
	}
	void refresh()
	{
		double m = H[0], s = 0.0, ln = std::log((double)total + 1.0);
		for (int j = 1; j < K; j++) m = std::max(m, H[j]);
		for (int j = 0; j < K; j++) { q[j] = std::exp(H[j] - m); s += q[j]; }
		for (int j = 0; j < K; j++)
		{
			q[j] /= s;
			score[j] = attempts[j] ? q[j] + C * std::sqrt(2.0 * ln / (double)attempts[j]) : 1e300;
		}
	}
	int select(const int * avail, int na)
	{
		int u = untried(avail, na);
		if (u >= 0) return u;
		int pool[K], np = 0;
		for (int i = 0; i < na; i++) if (q[avail[i]] >= bt) pool[np++] = avail[i];
		if (np == 0) for (int i = 0; i < na; i++) pool[np++] = avail[i];
		if (unif01() < eps) return pool[uniformIndex(np)];
		int bestId = pool[0];
		for (int i = 1; i < np; i++) if (score[pool[i]] > score[bestId]) bestId = pool[i];
		return bestId;
	}
	void update(int j, bool ok)
	{
		double r = ok ? 1.0 : 0.0;
		rbar += (r - rbar) / (double)(total + 1);
		for (int a = 0; a < K; a++) H[a] += alpha * (r - rbar) * ((a == j ? 1.0 : 0.0) - q[a]);
		AosPolicy::update(j, ok);
	}
};

inline AosPolicy * makeAosPolicy(int mode, unsigned seed)
{
	switch (mode)
	{
	case 1: return new UniformPolicy(seed);
	case 2: return new Ucb1Policy(seed);
	case 3: return new ThompsonPolicy(seed);
	case 4: return new ProbMatchingPolicy(seed);
	case 5: return new HgUcbPolicy(seed);
	default: return NULL;
	}
}

#endif
