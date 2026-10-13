"""Offline gate for tree drafting at the MTP -> base level: tokens kept per base pass, chain vs tree.

For each held-out position and MTP step d (the chain run along the real text, as mtp.py evaluates it), draws up to W
drafts from the MTP's sampler distribution q without replacement and accepts them by recursive rejection sampling
against the base's sampler distribution p (exact multi-draft speculative sampling). That gives a_d(w): the chance that
one of the first w siblings is accepted at depth d. With depth-wise independence (as mtp.py's kept@ assumes), a
dynamic program then finds the best tree for each node budget (Sequoia-style) and its expected accepted drafts; a base
pass also yields one token of its own, so tokens per pass = 1 + E.

  python tree_accept.py [--data E:/shoehorn/mtp/corpus.bin] [--held 100] [--depth 6] [--width 8] [--mc 32] [--load x.pt] [--device cpu]
"""
import argparse, time
import numpy as np
import torch
import mtp
from mtp import MTP, TOP_K, load_head_and_embd, read_data, sampler

DEV = mtp.DEV

CHUNK = 128


def dense(p_ids, p, q_ids, q):
    """p, q over their own top supports -> both dense over 2K slots (p's ids first, q's unmatched ids after)."""
    n, K = p.shape
    m = p_ids[:, :, None] == q_ids[:, None, :]                    # [n, Kp, Kq]
    hit = m.any(1)                                                 # q slot found in p
    slot = torch.where(hit, m.float().argmax(1), K + torch.arange(K, device=p.device)[None, :].expand(n, K))
    Q = torch.zeros(n, 2 * K, device=p.device).scatter_add_(1, slot, q)
    P = torch.cat([p, torch.zeros_like(p)], 1)
    return P, Q


def rrs(P, Q, W, S):
    """Recursive rejection sampling without replacement, S Monte Carlo draws per row -> acc [n, W]: the chance that one
    of the first w drafts was accepted."""
    P, Q = P.repeat_interleave(S, 0), Q.repeat_interleave(S, 0)
    done = torch.zeros(P.shape[0], dtype=torch.bool, device=P.device)
    out = []
    for _ in range(W):
        live = Q.sum(1) > 1e-9
        qn = torch.where(live[:, None], Q / Q.sum(1, keepdim=True).clamp_min(1e-12), torch.ones_like(Q))
        x = torch.multinomial(qn, 1).squeeze(1)
        px, qx = P.gather(1, x[:, None]).squeeze(1), qn.gather(1, x[:, None]).squeeze(1)
        ok = live & ~done & (torch.rand_like(px) < (px / qx.clamp_min(1e-12)).clamp(max=1))
        done = done | ok
        out.append(done.float().view(-1, S).mean(1))
        # rejected: p <- norm(max(p - q, 0)); q loses x
        rej = live & ~done
        Pn = (P - qn).clamp_min(0)
        Pn = Pn / Pn.sum(1, keepdim=True).clamp_min(1e-12)
        P = torch.where(rej[:, None], Pn, P)
        Q = torch.where(live[:, None], Q.scatter(1, x[:, None], 0.0), Q)
    return torch.stack(out, 1)


def collect(model, head, emb, seqs, D, W, S):
    """-> a [D, W]: mean acceptance with w siblings at depth d, along the real text."""
    tot, cnt = torch.zeros(D, W, device=DEV), 0
    t = lambda a: torch.from_numpy(np.array(a)).to(DEV)
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        for toks, h, ids, lg in seqs:
            n = len(toks) - 1 - D
            if n < 8:
                continue
            hin, pos, ctx, chain = t(h[:n]).float(), torch.arange(n, device=DEV), None, []
            for k in range(1, D + 1):
                ee = emb[torch.from_numpy(np.array(toks[k:k + n])).long()].to(DEV).float()
                hn, kv = model.step(hin, ee, pos + (k - 1), ctx, chain)
                if k == 1:
                    ctx = kv
                else:
                    chain.append(kv)
                pb, pb_ids = sampler(t(lg[k:k + n]), t(ids[k:k + n]).long())
                for c0 in range(0, n, CHUNK):
                    c1 = min(n, c0 + CHUNK)
                    top = (hn[c0:c1].to(torch.bfloat16) @ head.T).float().topk(TOP_K, -1)
                    q, q_ids = sampler(top.values, top.indices)
                    P, Q = dense(pb_ids[c0:c1], pb[c0:c1].float(), q_ids, q.float())
                    tot[k - 1] += rrs(P, Q, W, S).sum(0)
                hin = hn
            cnt += n
    return (tot / cnt).cpu().numpy()


def best_trees(a, N):
    """a [D, W] -> for budgets 1..N: (expected accepted drafts, tree as nested child lists) by dynamic programming.
    F[d][n]: best expectation below an accepted node at depth d with n nodes to place at depths d+1.."""
    D, W = a.shape
    r = np.diff(np.concatenate([np.zeros((D, 1)), a], 1), axis=1)  # r[d][i]: the (i+1)-th sibling is the accepted one
    F = [[0.0] * (N + 1) for _ in range(D + 1)]
    T = [[[] for _ in range(N + 1)] for _ in range(D + 1)]
    for d in range(D - 1, -1, -1):
        for n in range(1, N + 1):
            # knapsack over sibling ranks: G[i][m] = best with the first i siblings using m nodes
            G = [[-1.0] * (n + 1) for _ in range(W + 1)]
            C = [[None] * (n + 1) for _ in range(W + 1)]
            G[0][0] = 0.0
            for i in range(W):
                for m in range(n + 1):
                    if G[i][m] < 0:
                        continue
                    if G[i][m] > G[i + 1][m]:  # stop adding siblings (later ranks then stay empty)
                        pass
                    for s in range(1, n - m + 1):  # sibling i's subtree: itself + s - 1 below
                        v = G[i][m] + r[d][i] * (1 + F[d + 1][s - 1])
                        if v > G[i + 1][m + s]:
                            G[i + 1][m + s], C[i + 1][m + s] = v, (m, s)
            best, bi, bm = 0.0, 0, 0
            for i in range(1, W + 1):
                for m in range(1, n + 1):
                    if G[i][m] > best:
                        best, bi, bm = G[i][m], i, m
            F[d][n] = best
            kids, i, m = [], bi, bm
            while i > 0:
                pm, s = C[i][m]
                kids.append(T[d + 1][s - 1])
                i, m = i - 1, pm
            T[d][n] = kids[::-1]
    return F[0], T[0]


def size(t):
    return sum(1 + size(c) for c in t)


def shape(t):
    """Nodes per depth."""
    out, lvl = [], t
    while lvl:
        out.append(len(lvl))
        lvl = [c for k in lvl for c in k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'E:\shoehorn\mtp\corpus.bin')
    ap.add_argument('--held', type=int, default=100)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--width', type=int, default=8)
    ap.add_argument('--mc', type=int, default=32)
    ap.add_argument('--budget', type=int, default=64)
    ap.add_argument('--device', default='cuda', help='cpu: when the GPU is busy serving')
    ap.add_argument('--load', help='a layer saved by mtp.py --save (default: the stock 4-bit head)')
    a = ap.parse_args()
    global DEV
    DEV = mtp.DEV = a.device
    if DEV == 'cpu':
        torch.set_num_threads(16)
    t0 = time.time()
    torch.manual_seed(0)
    seqs = read_data(a.data)[-a.held:]
    head, emb = load_head_and_embd()
    model = MTP(bool(a.load)).to(DEV).eval()
    if a.load:
        model.w.load_state_dict({k: v.float() for k, v in torch.load(a.load).items()})
    acc = collect(model, head, emb, seqs, a.depth, a.width, a.mc)
    print(f'{a.load or "stock 4-bit head"}, {len(seqs)} held-out conversations, {time.time() - t0:.0f} s')
    print('a_d(w): chance one of the first w siblings is accepted')
    print('depth ' + ' '.join(f'  w={w}' for w in range(1, a.width + 1)))
    for d in range(a.depth):
        print(f'{d + 1:5d} ' + ' '.join(f'{x:6.3f}' for x in acc[d]))
    for D in (3, a.depth):
        c = np.cumprod(acc[:D, 0]).sum()
        print(f'chain of {D}: {c:.3f} drafts kept, {1 + c:.3f} tokens per base pass')
    F, T = best_trees(acc, a.budget)
    print('best tree (all depths): nodes -> drafts kept, tokens per base pass, nodes per depth')
    for n in (3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        if n <= a.budget:
            print(f'  {n:3d} -> {F[n]:.3f}, {1 + F[n]:.3f}  {shape(T[n])}')


if __name__ == '__main__':
    main()
