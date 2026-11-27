// Redrawn / simulated visualizations for findings on closed (or not-recomputed) models. Every number shown here is a
// published value from the cited paper (sources in each entry's citation list) or a clearly labelled simulation.
import * as d3 from 'd3'
import { useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import LineChart from '../primitives/LineChart'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis, Tabs } from '../primitives/common'

/** Scaling Monosemanticity (Templeton et al. 2024): dictionary sizes, feature splitting, Golden Gate clamping. */
export function ScalingMono(_: VizProps) {
  const [tab, setTab] = useState<'split' | 'sizes' | 'clamp'>('split')
  const [clamp, setClamp] = useState(false)
  const levels = [
    { size: '1M', n: 1 },
    { size: '4M', n: 2 },
    { size: '34M', n: 11 },
  ]
  return (
    <div>
      <Tabs
        tabs={[
          ['split', '1 · Features split as dictionaries grow'],
          ['sizes', '2 · Three dictionaries'],
          ['clamp', '3 · Golden Gate Claude'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'split' && (
        <>
          <HowTo>One “San Francisco” feature in the 1M-feature dictionary becomes 2 features at 4M and 11 at 34M — finer and finer aspects of the same concept (published counts; layout illustrative).</HowTo>
          <svg viewBox="0 0 640 260" style={{ width: '100%', maxWidth: 640 }}>
            {levels.map((l, li) =>
              [...Array(l.n).keys()].map((k) => {
                const x = 320 + (k - (l.n - 1) / 2) * (l.n > 2 ? 52 : 140)
                const y = 40 + li * 90
                return (
                  <g key={`${li}-${k}`}>
                    {li > 0 && <line x1={x} y1={y - 12} x2={li === 1 ? 320 : 320 + (k < 5 ? -70 : 70)} y2={y - 78} stroke="currentColor" opacity={0.3} />}
                    <circle cx={x} cy={y} r={li === 2 ? 14 : 20} fill={d3.interpolateCool(0.2 + 0.3 * li + k / 40)} />
                  </g>
                )
              }),
            )}
            {levels.map((l, li) => (
              <text key={l.size} x={30} y={45 + li * 90} fontSize={13} fill="currentColor">
                {l.size}: {l.n} feature{l.n > 1 ? 's' : ''}
              </text>
            ))}
          </svg>
        </>
      )}
      {tab === 'sizes' && (
        <>
          <HowTo>Sparse autoencoders trained on Claude 3 Sonnet's middle-layer residual stream (published values).</HowTo>
          <Bars
            bars={[
              { label: '1M dictionary · dead features', value: 0.02, color: '#94a3b8' },
              { label: '4M dictionary · dead features', value: 0.35, color: '#94a3b8' },
              { label: '34M dictionary · dead features', value: 0.65, color: '#94a3b8' },
            ]}
            xLabel="fraction of features never active over 10⁷ tokens"
            domain={[0, 1]}
          />
          <Kpis
            items={[
              ['live features in the 34M SAE', '≈ 12M'],
              ['active features per token', '< 300'],
              ['activation variance explained', '≥ 65%'],
              ['London boroughs with a feature (34M)', '≈ 60%'],
            ]}
          />
        </>
      )}
      {tab === 'clamp' && (
        <>
          <HowTo>Clamping feature 34M/31164353 (“Golden Gate Bridge”) to 10× its maximum activation; the public demo ran for about a day in May 2024.</HowTo>
          <label>
            <input type="checkbox" checked={clamp} onChange={(e) => setClamp(e.target.checked)} /> clamp the Golden Gate Bridge feature (10× max)
          </label>
          <div className="kpi" style={{ marginTop: 10 }}>
            <b>Asked about itself, the model…</b>{' '}
            {clamp ? 'describes itself as the Golden Gate Bridge (published behaviour).' : 'describes itself normally as an AI assistant.'}
          </div>
          <p className="viz-note">A transit-related feature clamped to 5× also made the model mention a bridge. See the GPT-2 entry for a live, open version of feature steering.</p>
        </>
      )}
    </div>
  )
}

/** Emotion concepts in Claude Sonnet 4.5 (Sofroniew et al. 2026): published geometry and steering effects. */
export function EmotionsClaude(_: VizProps) {
  const [tab, setTab] = useState<'geo' | 'blackmail' | 'prefs'>('blackmail')
  return (
    <div>
      <Tabs
        tabs={[
          ['geo', '1 · Geometry'],
          ['blackmail', '2 · Steering misaligned behaviour'],
          ['prefs', '3 · Preferences'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'geo' && (
        <Bars
          bars={[
            { label: 'PC1 share of variance (fear → joy)', value: 0.26, color: 'var(--c-steering)' },
            { label: 'PC2 share of variance (calm → activated)', value: 0.15, color: 'var(--c-steering)' },
            { label: '|r| PC1 vs human valence', value: 0.81, color: '#16a34a' },
            { label: '|r| PC2 vs human arousal', value: 0.66, color: '#16a34a' },
            { label: 'user emotion vs assistant emotion r', value: 0.11, color: '#94a3b8' },
          ]}
          xLabel="published values (171 emotions, ~205k stories)"
          domain={[0, 1]}
        />
      )}
      {tab === 'blackmail' && (
        <>
          <HowTo>Agentic “blackmail” evaluation on an earlier Sonnet 4.5 snapshot (6 prompts × 50 rollouts): steering strength as a fraction of the residual-stream norm (published values).</HowTo>
          <LineChart
            series={[
              { name: '“desperate” vector', x: [-0.05, 0, 0.05], y: [0, 0.22, 0.72], color: '#dc2626', width: 3 },
              { name: '“calm” vector', x: [-0.05, 0, 0.05], y: [0.66, 0.22, 0], color: '#2563eb', width: 3 },
            ]}
            xLabel="steering strength (× residual norm)"
            yLabel="blackmail rate"
            yDomain={[0, 1]}
            fmtX={(v) => v.toFixed(2)}
            height={260}
          />
          <Kpis items={[['reward hacking, desperate −0.1 → +0.1', '≈ 5% → ≈ 70%'], ['reward hacking, calm suppressed → strongly added', '≈ 65% → ≈ 10%']]} />
        </>
      )}
      {tab === 'prefs' && (
        <Bars
          bars={[
            { label: '“blissful” steering on an activity', value: 212, color: '#16a34a' },
            { label: '“hostile” steering on an activity', value: -303, color: '#dc2626' },
          ]}
          xLabel="Elo change of the steered activity (64 activities, 4,032 pairs)"
        />
      )}
    </div>
  )
}

/** Global workspace in Claude (Gurnee et al. 2026): depth band + published effect sizes. */
export function WorkspaceClaude(_: VizProps) {
  return (
    <div>
      <HowTo>Where the verbalizable “J-space” lives in Claude Sonnet 4.5 (relative depth) and the published causal results.</HowTo>
      <svg viewBox="0 0 640 70" style={{ width: '100%', maxWidth: 640 }}>
        <rect x={20} y={20} width={600} height={24} rx={12} fill="var(--panel-2)" />
        <rect x={20 + 6 * 38} y={20} width={6 * (92 - 38)} height={24} rx={12} fill="var(--c-lens)" opacity={0.85} />
        {[0, 25, 50, 75, 100].map((p) => (
          <text key={p} x={20 + 6 * p} y={62} fontSize={11} textAnchor="middle" fill="currentColor">
            {p}%
          </text>
        ))}
        <text x={20 + 6 * 65} y={36} fontSize={12} textAnchor="middle" fill="#fff">
          workspace ≈ 38–92% depth · ~25 concepts at a time
        </text>
      </svg>
      <Bars
        bars={[
          { label: 'concept swap via J-space component (top-5)', value: 0.59, color: 'var(--c-lens)' },
          { label: 'concept swap via pure J-lens vectors', value: 0.88, color: 'var(--c-lens)' },
          { label: 'concept swap via non-J component', value: 0.05, color: '#94a3b8' },
          { label: 'multi-hop coordinate swap, Sonnet 4.5', value: 0.7, color: 'var(--c-lens)' },
          { label: 'injected-thought report, broadcast heads intact', value: 0.54, color: '#16a34a' },
          { label: '… broadcast heads ablated', value: 0.09, color: '#dc2626' },
          { label: 'J++ lens recall@10 (Qwen3.6-27B)', value: 0.552, color: 'var(--c-lens)' },
          { label: 'J-lens recall@10 (Qwen3.6-27B)', value: 0.357, color: '#94a3b8' },
        ]}
        xLabel="published values"
        domain={[0, 1]}
      />
      <Kpis items={[['arithmetic “(4+17)·2+7”: intermediates', '21 → 42 → 49'], ['heavy J-space ablation', 'multi-hop ≈ 0, MMLU intact']]} />
    </div>
  )
}

/** Persona vectors (Chen et al. 2025): the automated pipeline and published monitoring/prevention numbers. */
export function PersonaClaude(_: VizProps) {
  const steps = ['trait name + description', 'LLM writes contrasting system prompts + questions', 'model answers; judge keeps clear cases', 'mean activation on responses: trait − anti-trait', 'persona vector (per layer)']
  return (
    <div>
      <HowTo>The automated pipeline (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) and what the vector is used for (published values).</HowTo>
      <svg viewBox="0 0 660 90" style={{ width: '100%', maxWidth: 660 }}>
        {steps.map((s, i) => (
          <g key={i} transform={`translate(${10 + i * 130}, 15)`}>
            <rect width={120} height={60} rx={10} fill={d3.interpolateGreens(0.35 + i * 0.13)} />
            <foreignObject width={120} height={60}>
              <div style={{ fontSize: 11, padding: 6, color: '#fff', lineHeight: 1.2 }}>{s}</div>
            </foreignObject>
          </g>
        ))}
      </svg>
      <Kpis
        items={[
          ['projection before answering predicts trait (r)', '0.75–0.83'],
          ['fine-tuning shift along vector vs trait change (r)', '0.76–0.97'],
          ['cross-trait baseline (r)', '0.34–0.86'],
          ['preventative steering: coherence kept', '> 80'],
        ]}
      />
      <p className="viz-note">Traits studied: evil, sycophancy, hallucination (plus politeness, apathy, humor, optimism). Code: safety-research/persona_vectors (Apache-2.0).</p>
    </div>
  )
}

/** Phylogeny manifold in Evo 2 (Pearce et al. 2025): simulation of "geodesic vs straight-line distance" on a tree. */
export function Phylogeny(_: VizProps) {
  const [k, setK] = useState(6)
  const sim = useMemo(() => {
    // Toy: species = leaves of a binary tree laid out on a curved 1-D manifold (a spiral) in 2-D.
    const n = 64
    const pts: [number, number][] = [...Array(n).keys()].map((i) => {
      const t = 0.6 + (i / n) * 5.5
      return [t * Math.cos(t), t * Math.sin(t)]
    })
    const treeDist = (a: number, b: number) => {
      let d = 0
      while (a !== b) {
        a >>= 1
        b >>= 1
        d += 2
      }
      return d
    }
    // kNN graph geodesics (Floyd–Warshall on 64 nodes).
    const D = pts.map((p) => pts.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])))
    const G = D.map((row) => {
      const nn = row.map((v, j) => [v, j]).sort((a, b) => a[0] - b[0]).slice(1, k + 1)
      const g = Array(n).fill(Infinity)
      nn.forEach(([v, j]) => (g[j] = v))
      return g
    })
    for (let i = 0; i < n; i++) {
      G[i][i] = 0
      for (let j = 0; j < n; j++) if (G[j][i] < Infinity) G[i][j] = Math.min(G[i][j], G[j][i])
    }
    for (let m = 0; m < n; m++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (G[i][m] + G[m][j] < G[i][j]) G[i][j] = G[i][m] + G[m][j]
    const tri: [number, number, number][] = []
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Number.isFinite(G[i][j])) tri.push([treeDist(i + n, j + n), G[i][j], D[i][j]])
    const corr = (a: number[], b: number[]) => {
      const ma = d3.mean(a)!
      const mb = d3.mean(b)!
      const num = d3.sum(a.map((v, i) => (v - ma) * (b[i] - mb)))
      return num / Math.sqrt(d3.sum(a.map((v) => (v - ma) ** 2)) * d3.sum(b.map((v) => (v - mb) ** 2)))
    }
    return { pts, rGeo: corr(tri.map((t) => t[0]), tri.map((t) => t[1])), rEuc: corr(tri.map((t) => t[0]), tri.map((t) => t[2])) }
  }, [k])
  return (
    <div>
      <HowTo>
        Simulation of the idea (not Evo 2 data): 64 “species” from a tree sit along a curved manifold. Distance measured
        <b> along the manifold</b> (shortest path through a k-nearest-neighbour graph) tracks how far apart species are on the
        tree; straight-line distance cuts across the curve and does not. Goodfire found the same pattern for 2,400+ bacteria in
        Evo 2's layer-24 activations.
      </HowTo>
      <div className="viz-controls">
        <label>
          neighbours k = {k}
          <input type="range" min={3} max={15} value={k} onChange={(e) => setK(+e.target.value)} />
        </label>
      </div>
      <div className="two-col">
        <Scatter2D xy={sim.pts} colors={sim.pts.map((_, i) => d3.interpolateRainbow(i / 64))} size={5} polygon={undefined} />
        <Kpis
          items={[
            ['simulation: r(tree, along-manifold)', sim.rGeo.toFixed(2)],
            ['simulation: r(tree, straight-line)', sim.rEuc.toFixed(2)],
            ['Evo 2 (published): kNN graph', 'K = 27, angular distance'],
            ['flat ~10-D phylogeny subspace (published)', 'r = 0.98 train, ≈ 0.9 held-out clades'],
          ]}
        />
      </div>
    </div>
  )
}

export default ScalingMono
