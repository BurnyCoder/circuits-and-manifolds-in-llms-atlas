// About: what the atlas is, how data is produced, fidelity levels, licensing of quotes/figures.
import { FIDELITY } from '../lib/labels'

export default function About() {
  return (
    <div className="prose">
      <h1>About this atlas</h1>
      <p>
        The Atlas of Circuits &amp; Manifolds collects the most striking mechanisms found inside neural networks by
        mechanistic-interpretability research and makes each one explorable. It is maintained by Libor “Burny” Burian.
        Source code, data pipeline and reports:{' '}
        <a href="https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas" target="_blank" rel="noopener">
          github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas
        </a>
        .
      </p>
      <h2>Where the numbers come from</h2>
      <ul>
        {Object.entries(FIDELITY).map(([k, f]) => (
          <li key={k}>
            <b>
              {f.icon} {f.label}:
            </b>{' '}
            {f.help}
          </li>
        ))}
      </ul>
      <p>
        Recomputed entries are produced by <code>uv run python -m atlas.run</code>, which writes one JSON file per entry.
        Numbers in the prose are filled in from those files, never typed by hand, and every entry shows the published
        value next to ours with a pass criterion fixed in advance.
      </p>
      <h2>Quotes, figures and licenses</h2>
      <p>
        Verbatim quotes are limited to one short excerpt (under 15 words) per entry, with attribution. All other text is
        our own paraphrase with links to the source section. Figures are redrawn from our own data or from published
        numbers; we do not copy figures from papers or sites whose license does not allow it.
      </p>
    </div>
  )
}
