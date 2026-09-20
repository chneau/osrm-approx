// S2: Pruned Landmark Labeling (PLL) hub-label oracle over a sparse road graph.
//
// Reads a graph (n, m, then m x (u32 v32 dist f32 dur f32), undirected), builds hub
// labels for one weight (distance or duration), and writes them. A query is a
// two-pointer intersection of the two nodes' label lists -- no graph traversal.
//
// Build:  gcc -O2 -o experiments/pll experiments/pll.c -lm
// Run:    experiments/pll experiments/s2_graph.bin <0=dist|1=dur> experiments/s2.labels
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { int hub; float dist; } Lab;
typedef struct { float d; int v; } HN;

static int n, m;
static int *head, *to, *nxt;
static float *wt;
static Lab **labels; static int *lcnt, *lcap;
static float *dist;
static HN *heap; static int hn;

static void hpush(float d, int v){
    int i = hn++; heap[i].d = d; heap[i].v = v;
    while (i) { int p = (i-1)/2; if (heap[p].d <= heap[i].d) break;
        HN t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p; }
}
static HN hpop(void){
    HN r = heap[0]; heap[0] = heap[--hn]; int i = 0;
    for (;;) { int l = 2*i+1, rr = l+1, s = i;
        if (l < hn && heap[l].d < heap[s].d) s = l;
        if (rr < hn && heap[rr].d < heap[s].d) s = rr;
        if (s == i) break; HN t = heap[s]; heap[s] = heap[i]; heap[i] = t; i = s; }
    return r;
}
static void ladd(int u, int hub, float d){
    if (lcnt[u] == lcap[u]) { lcap[u] = lcap[u] ? lcap[u]*2 : 8;
        labels[u] = realloc(labels[u], (size_t)lcap[u]*sizeof(Lab)); }
    labels[u][lcnt[u]].hub = hub; labels[u][lcnt[u]].dist = d; lcnt[u]++;
}
// labels are appended in increasing hub rank -> sorted by hub, two-pointer.
static float query(int a, int b){
    int i = 0, j = 0; float best = 1e30f;
    while (i < lcnt[a] && j < lcnt[b]) {
        int ha = labels[a][i].hub, hb = labels[b][j].hub;
        if (ha == hb) { float s = labels[a][i].dist + labels[b][j].dist;
            if (s < best) best = s; i++; j++; }
        else if (ha < hb) i++; else j++;
    }
    return best;
}

int main(int argc, char **argv){
    if (argc < 4) { fprintf(stderr, "usage: %s graph.bin weight(0=dist,1=dur) out.labels\n", argv[0]); return 2; }
    int weight = atoi(argv[2]);
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("graph"); return 1; }
    if (fread(&n, 4, 1, f) != 1 || fread(&m, 4, 1, f) != 1) return 1;
    fprintf(stderr, "[pll] n=%d m=%d weight=%s\n", n, m, weight ? "dur" : "dist");

    int *eu = malloc((size_t)m*4), *ev = malloc((size_t)m*4);
    float *ed = malloc((size_t)m*4), *et = malloc((size_t)m*4);
    for (int i = 0; i < m; i++) {
        if (fread(&eu[i],4,1,f)!=1||fread(&ev[i],4,1,f)!=1
            ||fread(&ed[i],4,1,f)!=1||fread(&et[i],4,1,f)!=1) return 1;
    }
    fclose(f);

    // CSR (undirected)
    int *deg = calloc(n, sizeof(int));
    for (int i = 0; i < m; i++) { if (eu[i]<0||eu[i]>=n||ev[i]<0||ev[i]>=n) continue; deg[eu[i]]++; deg[ev[i]]++; }
    head = calloc(n+1, sizeof(int));
    for (int i = 0; i < n; i++) head[i+1] = head[i] + deg[i];
    to = malloc((size_t)head[n]*4); nxt = malloc((size_t)head[n]*4); wt = malloc((size_t)head[n]*4);
    int *cur = malloc((size_t)n*4); memcpy(cur, head, (size_t)n*4);
    for (int i = 0; i < m; i++) {
        if (eu[i]<0||eu[i]>=n||ev[i]<0||ev[i]>=n) continue;
        float w = weight ? et[i] : ed[i];
        to[cur[eu[i]]] = ev[i]; wt[cur[eu[i]]] = w; cur[eu[i]]++;
        to[cur[ev[i]]] = eu[i]; wt[cur[ev[i]]] = w; cur[ev[i]]++;
    }
    free(cur); free(eu); free(ev); free(ed); free(et);

    // Order by degree descending (rank position v -> rank).
    int *order = malloc((size_t)n*4), *rank = malloc((size_t)n*4);
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++)
        if (deg[order[j]] > deg[order[i]]) { int t=order[i]; order[i]=order[j]; order[j]=t; }
    for (int i = 0; i < n; i++) rank[order[i]] = i;

    labels = calloc(n, sizeof(Lab*)); lcnt = calloc(n, sizeof(int)); lcap = calloc(n, sizeof(int));
    dist = malloc((size_t)n*sizeof(float));
    int maxedges = head[n];
    heap = malloc((size_t)(maxedges + n + 8)*sizeof(HN));

    for (int ri = 0; ri < n; ri++) {
        int r = order[ri];
        for (int i = 0; i < n; i++) dist[i] = 1e30f;
        dist[r] = 0; hn = 0; hpush(0, r);
        while (hn) {
            HN top = hpop(); float d = top.d; int u = top.v;
            if (d > dist[u]) continue;
            if (query(r, u) <= d) continue;     // pruned
            ladd(u, rank[r], d);
            for (int e = head[u]; e < head[u+1]; e++) {
                float nd = d + wt[e];
                if (nd < dist[to[e]]) { dist[to[e]] = nd; hpush(nd, to[e]); }
            }
        }
        if (ri % 2000 == 0) fprintf(stderr, "[pll]   root %d/%d  labels=%lld\n", ri, n, (long long)0);
    }

    long long total = 0; int mx = 0;
    for (int i = 0; i < n; i++) { total += lcnt[i]; if (lcnt[i] > mx) mx = lcnt[i]; }
    fprintf(stderr, "[pll] labels: total=%lld avg=%.1f max=%d (%.1f MB raw)\n",
            total, (double)total/n, mx, (double)total*8/1e6);

    FILE *o = fopen(argv[3], "wb");
    fwrite(&n, 4, 1, o);
    for (int i = 0; i < n; i++) {
        fwrite(&lcnt[i], 4, 1, o);
        fwrite(labels[i], sizeof(Lab), lcnt[i], o);
    }
    fclose(o);
    fprintf(stderr, "[pll] wrote %s\n", argv[3]);
    return 0;
}
