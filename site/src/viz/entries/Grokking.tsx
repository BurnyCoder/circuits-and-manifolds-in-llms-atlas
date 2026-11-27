// Grokking entry: training curves (3 seeds + no-weight-decay control), the embedding's Fourier spectrum over training
// with a scrubber, embedding circles for each key frequency, periodic neuron heatmaps, and the Fourier ablation.
import * as d3 from 'd3'
import { useMemo, useState } from 'react'
import { useData } from '../../lib/data'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import Heatmap, { Legend } from '../primitives/Heatmap'
import LineChart from '../primitives/LineChart'
import { HowTo, Kpis, Tabs, seqScale } from '../primitives/common'

interface Curves {
  epoch: number[]
  train_loss: number[]
  test_loss: number[]
  train_acc: number[]
  test_acc: number[]
  wnorm: number[]
}
interface Run {
  seed: number
  curves: Curves
  key_freqs: number[]
  frac_periodic: number
  we_norms: number[]
  n_freq_90: number
  ablate: Record<string, number>
}
interface D {
  runs: Run[]
  wd0: Curves
  snaps: number[][]
  snap_every: number
  circles: Record<string, [number, number][]>
  neurons: { neuron: number; freq: number; fve: number; grid: number[][] }[]
}

const P = 113

export default function Grokking({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'curves' | 'spectrum' | 'circles' | 'neurons' | 'ablate' | 'runs'>('curves')
  const [seed, setSeed] = useState(0)
  const [snap, setSnap] = useState(d.snaps.length - 1)
  const keys = Object.keys(d.circles)
  const [freq, setFreq] = useState(keys[0])
  const [ordered, setOrdered] = useState(false)
  const run = d.runs[seed]
  const keySet = new Set(d.runs[0].key_freqs)
  const spec = useMemo(() => d.snaps[0].map((_, k) => d.snaps.map((s) => s[k])), [d.snaps]) // [freq][snapshot]
  const specMax = Math.max(...d.snaps.flat())
  const specColor = useMemo(() => seqScale(specMax, d3.interpolateMagma), [specMax])
  const neuronColor = useMemo(() => seqScale(255, d3.interpolateInferno), [])

  return (
    <div>
      <Tabs
        tabs={[
          ['curves', '1 · Memorize, then grok'],
          ['spectrum', '2 · Fourier spectrum over training'],
          ['circles', '3 · Numbers on circles'],
          ['neurons', '4 · Periodic neurons'],
          ['ablate', '5 · Ablate the frequencies'],
          ['runs', '6 · 22 runs, 11 seeds'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'curves' && (
        <>
          <HowTo>
            A one-layer transformer learns (a + b) mod 113 from 30% of all pairs. Training loss drops fast (memorization);
            test loss stays high for thousands of epochs and then suddenly collapses (generalization). The dashed line is
            the same model trained <b>without weight decay</b>.
          </HowTo>
          <div className="viz-controls">
            {d.runs.map((r, i) => (
              <label key={r.seed}>
                <input type="radio" checked={seed === i} onChange={() => setSeed(i)} /> seed {r.seed}
              </label>
            ))}
          </div>
          <LineChart
            series={[
              { name: 'train loss', x: run.curves.epoch, y: run.curves.train_loss, color: '#2563eb' },
              { name: 'test loss', x: run.curves.epoch, y: run.curves.test_loss, color: '#dc2626' },
              { name: 'test loss, no weight decay', x: d.wd0.epoch, y: d.wd0.test_loss, color: '#dc2626', dash: '5 4', width: 1.5 },
            ]}
            yLog
            xLabel="epoch (full-batch steps)"
            yLabel="cross-entropy (log scale)"
          />
          <LineChart
            series={[
              { name: 'train accuracy', x: run.curves.epoch, y: run.curves.train_acc, color: '#2563eb' },
              { name: 'test accuracy', x: run.curves.epoch, y: run.curves.test_acc, color: '#dc2626' },
              { name: 'test accuracy, no weight decay', x: d.wd0.epoch, y: d.wd0.test_acc, color: '#dc2626', dash: '5 4', width: 1.5 },
            ]}
            yDomain={[0, 1]}
            xLabel="epoch"
            yLabel="accuracy"
            height={220}
          />
          <Kpis
            items={[
              ['key frequencies (this seed)', run.key_freqs.join(', ')],
              ['neurons with one dominant frequency', `${(run.frac_periodic * 100).toFixed(0)}%`],
              ['frequencies holding 90% of ‖W_E‖²', run.n_freq_90],
            ]}
          />
        </>
      )}
      {tab === 'spectrum' && (
        <>
          <HowTo>
            Rows are frequencies k = 1…56; columns are training snapshots every {d.snap_every} epochs (seed 0). Color = how
            much of the token embedding lies on cos/sin(2πk·a/113). Drag the slider: the spectrum goes from noise to a few
            bright key frequencies (marked ★) as the model groks, while weight decay wipes out the rest.
          </HowTo>
          <Heatmap
            values={spec}
            color={specColor}
            cell={Math.max(3, Math.floor(640 / d.snaps.length))}
            yLabels={spec.map((_, k) => `${keySet.has(k + 1) ? '★' : ''}${k + 1}`)}
            xLabels={d.snaps.map((_, i) => `${(i * d.snap_every) / 1000}k`)}
            xTitle="epoch"
            yTitle="frequency k"
            selected={`0,${snap}`}
            onCellClick={(_, c) => setSnap(c)}
            tooltip={(r, c, v) => `k=${r + 1}, epoch ${c * d.snap_every}: ${v.toFixed(2)}`}
          />
          <Legend color={specColor} lo={0} hi={specMax} label="‖W_E‖ on frequency k" />
          <div className="viz-controls">
            <label>
              epoch {snap * d.snap_every}
              <input type="range" min={0} max={d.snaps.length - 1} value={snap} onChange={(e) => setSnap(+e.target.value)} />
            </label>
          </div>
          <Bars
            bars={d.snaps[snap]
              .map((v, k) => ({ label: `k = ${k + 1}${keySet.has(k + 1) ? ' ★' : ''}`, value: v, color: keySet.has(k + 1) ? 'var(--c-manifold)' : '#94a3b8' }))
              .sort((a, b) => b.value - a.value)
              .slice(0, 10)}
            xLabel={`top-10 frequencies at epoch ${snap * d.snap_every}`}
            domain={[0, specMax]}
          />
        </>
      )}
      {tab === 'circles' && (
        <>
          <HowTo>
            Project the 113 number embeddings onto the two directions the model uses for one key frequency. Each number a
            lands at angle ≈ 2πk·a/113: <b>a circle</b>. Lines connect a → a+1 (a star pattern, step k); tick “sort by
            angle” to connect neighbors on the circle instead. Color = value of a.
          </HowTo>
          <div className="viz-controls">
            {keys.map((k) => (
              <label key={k}>
                <input type="radio" checked={freq === k} onChange={() => setFreq(k)} /> k = {k}
              </label>
            ))}
            <label>
              <input type="checkbox" checked={ordered} onChange={(e) => setOrdered(e.target.checked)} /> sort by angle
            </label>
          </div>
          <Circle pts={d.circles[freq]} k={+freq} ordered={ordered} />
        </>
      )}
      {tab === 'neurons' && (
        <>
          <HowTo>
            Activation of single MLP neurons (at the “=” token) for every input pair: a on the vertical axis, b on the
            horizontal. Each neuron is a clean wave in a and b with one frequency: the model builds cos/sin(w(a+b)) from
            products of these waves.
          </HowTo>
          <div className="two-col">
            {d.neurons.map((n) => (
              <div key={n.neuron}>
                <h3>
                  neuron {n.neuron} · frequency {n.freq} · {(n.fve * 100).toFixed(0)}% variance from that frequency
                </h3>
                <Heatmap values={n.grid} color={neuronColor} cell={3} xTitle="b" yTitle="a" tooltip={(r, c, v) => `a=${r}, b=${c}: ${(v / 255).toFixed(2)} (scaled)`} />
              </div>
            ))}
          </div>
        </>
      )}
      {tab === 'runs' && <Runs />}
      {tab === 'ablate' && (
        <>
          <HowTo>
            Causal test (seed 0): delete the key-frequency components from the token embedding and measure test accuracy;
            compare with deleting the same number of random other frequencies (mean of 10 draws).
          </HowTo>
          <Bars
            bars={Object.entries(d.runs[0].ablate).map(([k, v], i) => ({ label: k, value: v, color: ['#16a34a', '#dc2626', '#64748b'][i] }))}
            xLabel="test accuracy"
            domain={[0, 1]}
          />
        </>
      )}
    </div>
  )
}

interface Run22 {
  id: string
  group: string
  key_freqs: number[]
  diffuse: boolean
  frac_neurons_085: number | null
  test95: number | null
  train99: number | null
  final_test_acc: number
  curve: { step: number[]; test_acc: number[]; train_acc: number[] }
}
const GROUP_COLOR: Record<string, string> = { wd1: '#16a34a', split0: '#2563eb', wd0: '#dc2626', rand: '#64748b' }
const GROUP_LABEL: Record<string, string> = {
  wd1: 'weight decay 1 (11 seeds)',
  split0: "paper's split, other inits (3)",
  wd0: 'no weight decay (5)',
  rand: 'random labels (3)',
}

/** Companion project's 22 runs (github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer). */
function Runs() {
  const { data, error } = useData('grokking-runs.json')
  if (error) return <p className="error">{error}</p>
  if (!data) return <p>Loading 22 runs…</p>
  const runs = (data as unknown as { runs: Run22[] }).runs
  const groups = [...new Set(runs.map((r) => r.group))]
  return (
    <>
      <HowTo>
        22 training runs from the companion project{' '}
        <a href="https://burnycoder.github.io/visualizing-grokking-modular-addition-in-transformer/" target="_blank" rel="noopener">
          Grokking, Visualized
        </a>{' '}
        (models on{' '}
        <a href={String(data.source)} target="_blank" rel="noopener">
          Hugging Face
        </a>
        ), all with the paper's settings: every weight-decay seed groks, but each picks its <b>own</b> key frequencies; runs
        without weight decay or with random labels never generalize. Curves show test accuracy.
      </HowTo>
      <LineChart
        series={runs.map((r) => ({ name: r.id, x: r.curve.step.map((s) => Math.max(s, 1)), y: r.curve.test_acc, color: GROUP_COLOR[r.group] ?? '#999', width: 1.4 }))}
        xLog
        yDomain={[0, 1]}
        xLabel="optimizer step (log scale)"
        yLabel="test accuracy"
        height={300}
      />
      <div className="viz-controls">
        {groups.map((g) => (
          <span key={g} className="tag" style={{ background: GROUP_COLOR[g], color: '#fff' }}>
            {GROUP_LABEL[g] ?? g}
          </span>
        ))}
      </div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>run</th>
              <th>key frequencies</th>
              <th>single-freq. neurons</th>
              <th>train ≥ 99% at step</th>
              <th>test ≥ 95% at step</th>
              <th>final test acc.</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td style={{ color: GROUP_COLOR[r.group] }}>{r.id}</td>
                <td>{r.diffuse ? <i>none (diffuse spectrum)</i> : r.key_freqs.join(', ')}</td>
                <td>{r.frac_neurons_085 == null ? '—' : `${(r.frac_neurons_085 * 100).toFixed(0)}%`}</td>
                <td>{r.train99 ?? '—'}</td>
                <td>{r.test95 ?? 'never'}</td>
                <td>{(r.final_test_acc * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function Circle({ pts, k, ordered }: { pts: [number, number][]; k: number; ordered: boolean }) {
  const S = 380
  const ext = Math.max(...pts.flat().map(Math.abs)) * 1.1
  const s = d3.scaleLinear().domain([-ext, ext]).range([10, S - 10])
  const idx = ordered ? [...pts.keys()].sort((a, b) => ((a * k) % P) - ((b * k) % P)) : [...pts.keys()]
  const path = d3.line<number>().x((i) => s(pts[i][0])).y((i) => s(pts[i][1]))([...idx, idx[0]]) ?? ''
  return (
    <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: S }}>
      <path d={path} fill="none" stroke="currentColor" opacity={0.25} />
      {pts.map(([x, y], a) => (
        <g key={a}>
          <circle cx={s(x)} cy={s(y)} r={5} fill={d3.interpolateRainbow(a / P)} stroke="var(--panel)" strokeWidth={0.8}>
            <title>a = {a}</title>
          </circle>
          {a % 10 === 0 && (
            <text x={s(x) + 7} y={s(y) + 4} fontSize={10} fill="currentColor">
              {a}
            </text>
          )}
        </g>
      ))}
    </svg>
  )
}
