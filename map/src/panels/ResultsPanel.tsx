import { formatDate, formatPercent } from '../data/format'
import type { PreparedNetwork } from '../data/network'
import { DELAY_BANDS } from '../data/replay'
import { BarList, Dumbbell, Heatmap, LineChart, StackedBars, TableView } from '../components/charts'
import { lineName, SERIES, seconds } from '../data/labels'
import { LineTag } from '../components/LineTag'
import type { ModelScore, Period, Results } from '../types'

const HEADLINE = 'hist_gradient_boosting_change'
const CANDIDATE = 'hist_gradient_boosting_run_time'
const PERSISTENCE = 'baseline_persistence'
const LOOKUP = 'baseline_run_time'

/** Consistent identities across every chart: our model, the rule of thumb, the lookup. */
function colorOf(id: string) {
  if (id === HEADLINE) return SERIES.model
  if (id === PERSISTENCE) return SERIES.persistence
  if (id === LOOKUP) return SERIES.lookup
  return undefined
}

const shortDate = (date: string) => formatDate(date.length === 8 ? `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6)}` : date, 'short')
const pct = (value: number) => `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`
const score = (period: Period, id: string) => period.regression.find((row) => row.id === id)

export function ResultsPanel({ network, onHighlightLine, onPeriod, periodId, results }: {
  network: PreparedNetwork
  onHighlightLine: (line: string | undefined) => void
  onPeriod: (id: string) => void
  periodId: string
  results: Results
}) {
  const period = results.periods.find((p) => p.id === periodId) ?? results.periods[0]
  if (!period) {
    return (
      <header className="panel-head">
        <h1>Results</h1>
        <p>No results exported yet. Run <code>python -m mbta_ds.cli export-map</code> after training.</p>
      </header>
    )
  }
  return (
    <>
      <header className="panel-head">
        <h1>Results</h1>
        <p>How well a train’s delay at its next stop can be predicted, and where the T runs late. Every number is from a test period the models never trained on.</p>
      </header>

      <AcrossPeriods results={results} />

      <div className="period-bar">
        {results.periods.length > 1 ? (
          <div aria-label="Period" className="segmented" role="radiogroup">
            {results.periods.map((p) => (
              <button aria-checked={p.id === period.id} key={p.id} onClick={() => onPeriod(p.id)} role="radio" type="button">{p.label}</button>
            ))}
          </div>
        ) : <h2>{period.label}</h2>}
        <p className="hint">
          {period.note} Tested on {shortDate(period.split.test_dates[0])} to {shortDate(period.split.test_dates[1])}.
        </p>
      </div>

      <Headline period={period} />
      <Methods period={period} />
      <Horizons period={period} />
      <Backtest period={period} />
      {period.bandsByLine && period.bands ? <Bands network={network} onHighlightLine={onHighlightLine} period={period} /> : null}
      {period.lateByHour && period.hours ? <ByHour network={network} period={period} /> : null}
      <ByLine network={network} onHighlightLine={onHighlightLine} period={period} />
      {period.importance ? <Importance period={period} /> : null}
      {period.warning ? <EarlyWarning period={period} /> : null}
    </>
  )
}

function AcrossPeriods({ results }: { results: Results }) {
  const rows = results.periods.flatMap((p) => {
    const model = score(p, HEADLINE)
    const guess = score(p, PERSISTENCE)
    return model && guess ? [{ from: guess.mae, key: p.id, label: p.label, to: model.mae }] : []
  })
  if (rows.length < 2) return null
  return (
    <section className="panel-section">
      <h2>Every period</h2>
      <p className="hint">Average next-stop error. The model was designed on spring and winter; July–September checks a later period. The report records subsequent corrections.</p>
      <Dumbbell format={seconds} rows={rows} />
      <TableView columns={['Period', 'Rule of thumb', 'Model', 'Smaller by']} rows={rows.map((r) => [r.label, seconds(r.from), seconds(r.to), pct(1 - r.to / r.from)])} />
    </section>
  )
}

function Headline({ period }: { period: Period }) {
  const model = score(period, HEADLINE)
  const guess = score(period, PERSISTENCE)
  const clf = period.classification.find((row) => row.id === 'hist_gradient_boosting')
  if (!model || !guess) return null
  return (
    <section className="panel-section">
      <div className="hero">
        <p className="hero-label">Our model’s average error, one stop ahead</p>
        <p className="hero-value">{seconds(model.mae)}</p>
        <p className="hero-note">
          The rule of thumb “it stays as late as it is” is off by {seconds(guess.mae)}: the model’s error is {pct(1 - model.mae / guess.mae)} smaller.
        </p>
      </div>
      <dl className="stat-grid is-three">
        {period.lateShare !== undefined ? <div><dt>Arrivals late for riders</dt><dd>{pct(period.lateShare)}</dd><small>5+ min longer wait than planned</small></div> : null}
        {period.arrivals !== undefined ? <div><dt>Arrivals analysed</dt><dd>{(period.arrivals / 1e6).toFixed(1)}M</dd><small>{period.window[0] ? `${shortDate(period.window[0])} – ${shortDate(period.window[1]!)}` : ''}</small></div> : null}
        {clf ? <div><dt>Late or not</dt><dd>F1 {clf.f1.toFixed(2)}</dd><small>ROC-AUC {clf.rocAuc.toFixed(3)}</small></div> : null}
      </dl>
    </section>
  )
}

function Methods({ period }: { period: Period }) {
  const rows = [...period.regression].sort((a, b) => a.mae - b.mae)
  // The two "on time" baselines err by minutes and would flatten every other bar.
  const shown = rows.filter((row) => row.mae < 200)
  const note = (row: ModelScore) => (row.id === HEADLINE ? 'fixed in advance as the headline'
    : row.id === CANDIDATE ? 'a candidate: designed after test scores were seen' : undefined)
  return (
    <section className="panel-section">
      <h2>Every method, one stop ahead</h2>
      <p className="hint">Average error of each method’s next-stop prediction, lower is better.</p>
      <BarList
        bars={shown.map((row) => ({ color: colorOf(row.id), key: row.id, label: row.label, note: note(row), value: row.mae }))}
        format={seconds}
      />
      <p className="hint">Not shown: {rows.filter((row) => row.mae >= 200).map((row) => `${row.label.toLowerCase()} (${Math.round(row.mae)} s)`).join(', ')}.</p>
      <TableView columns={['Method', 'Mean error', 'RMSE', 'R²']} rows={rows.map((r) => [r.label, seconds(r.mae), seconds(r.rmse), r.r2.toFixed(3)])} />
    </section>
  )
}

function Horizons({ period }: { period: Period }) {
  const h = period.horizons
  const [near, far] = [h[0], h[h.length - 1]]
  const caption = near.runTime < near.model && far.model < far.runTime
    ? `One stop ahead the running-time lookup is better (${seconds(near.runTime)} against ${seconds(near.model)}); by ${far.stops} stops the model is (${seconds(far.model)} against ${seconds(far.runTime)}).`
    : `Average error by how many stops ahead the prediction is made.`
  return (
    <section className="panel-section">
      <h2>Further ahead</h2>
      <p className="hint">{caption}</p>
      <LineChart
        format={(v) => `${Math.round(v)} s`}
        series={[
          { color: SERIES.persistence, key: 'persistence', label: 'Rule of thumb', values: h.map((r) => r.persistence) },
          { color: SERIES.lookup, key: 'lookup', label: 'Running-time lookup', values: h.map((r) => r.runTime) },
          { color: SERIES.model, key: 'model', label: 'Our model', values: h.map((r) => r.model) },
        ]}
        x={h.map((r) => r.stops)}
        xFormat={(v) => String(v)}
        xLabel="stops ahead"
      />
      <TableView columns={['Stops ahead', 'Rule of thumb', 'Lookup', 'Model', 'Arrivals']} rows={h.map((r) => [r.stops, seconds(r.persistence), seconds(r.runTime), seconds(r.model), r.n.toLocaleString('en-US')])} />
    </section>
  )
}

function Backtest({ period }: { period: Period }) {
  const b = period.backtest
  if (b.length < 2) return null
  return (
    <section className="panel-section">
      <h2>Fortnight by fortnight</h2>
      <p className="hint">Retrained before each two-week window and scored on it, so one lucky test period cannot carry the result.</p>
      <LineChart
        format={(v) => `${Math.round(v)} s`}
        height={200}
        series={[
          { color: SERIES.persistence, key: 'persistence', label: 'Rule of thumb', values: b.map((r) => r.persistence) },
          { color: SERIES.lookup, key: 'lookup', label: 'Running-time lookup', values: b.map((r) => r.runTime) },
          { color: SERIES.model, key: 'model', label: 'Our model', values: b.map((r) => r.model) },
        ]}
        x={b.map((_, i) => i)}
        xFormat={(i) => shortDate(b[i].from).replace(/ \d{4}$/, '')}
        xLabel="window starting"
      />
      <TableView columns={['From', 'To', 'Rule of thumb', 'Lookup', 'Model', 'Candidate']} rows={b.map((r) => [shortDate(r.from), shortDate(r.to), seconds(r.persistence), seconds(r.runTime), seconds(r.model), seconds(r.candidate)])} />
    </section>
  )
}

function Bands({ network, onHighlightLine, period }: { network: PreparedNetwork; onHighlightLine: (line: string | undefined) => void; period: Period }) {
  const rows = [...period.bandsByLine!].reverse()
  return (
    <section className="panel-section">
      <h2>How late trains run</h2>
      <p className="hint">Share of each line’s arrivals by how much longer than planned riders waited. The figure at the right is the share 5+ min late.</p>
      <StackedBars
        // "On time" recedes to a neutral track: the late bands are the subject, and a
        // near-white block the width of the bar would outshout them.
        colors={DELAY_BANDS.map((band, i) => (i === 0 ? 'var(--surface-3)' : band.color))}
        labels={DELAY_BANDS.map((band) => band.label)}
        onHover={onHighlightLine}
        rows={rows.map((row) => ({ key: row.line, label: <LineTag id={row.line} line={network.lineById.get(row.line)} />, shares: row.shares }))}
      />
      <TableView columns={['Line', ...DELAY_BANDS.map((band) => band.label)]} rows={rows.map((row) => [row.line, ...row.shares.map(pct)])} />
    </section>
  )
}

function ByHour({ network, period }: { network: PreparedNetwork; period: Period }) {
  const hour = (h: number) => `${String(h).padStart(2, '0')}:00`
  return (
    <section className="panel-section">
      <h2>When it’s worst</h2>
      <p className="hint">Share of arrivals 5+ min late for riders, by scheduled hour. Blank where a line had too few trains that hour.</p>
      <Heatmap
        colFormat={hour}
        cols={period.hours!}
        format={pct}
        rows={period.lateByHour!.map((row) => ({ key: row.line, label: <LineTag id={row.line} line={network.lineById.get(row.line)} />, values: row.shares }))}
      />
      <TableView columns={['Line', ...period.hours!.map(hour)]} rows={period.lateByHour!.map((row) => [row.line, ...row.shares.map((v) => (v === null ? '' : pct(v)))])} />
    </section>
  )
}

function ByLine({ network, onHighlightLine, period }: { network: PreparedNetwork; onHighlightLine: (line: string | undefined) => void; period: Period }) {
  const rows = [...period.byLine].sort((a, b) => b.mae - a.mae)
  return (
    <section className="panel-section" onPointerLeave={() => onHighlightLine(undefined)}>
      <h2>Model error by line</h2>
      <p className="hint">Average next-stop error on each line; {lineName(network.lineById.get(rows[0].line), rows[0].line)} is the hardest to predict.</p>
      <div onPointerOver={(event) => onHighlightLine((event.target as HTMLElement).closest<HTMLElement>('[data-line]')?.dataset.line)}>
        <BarList bars={rows.map((row) => ({ color: SERIES.model, key: row.line, label: <span data-line={row.line}><LineTag id={row.line} line={network.lineById.get(row.line)} /></span>, value: row.mae }))} format={seconds} />
      </div>
      <TableView columns={['Line', 'Mean error', 'Median error', 'Arrivals']} rows={rows.map((r) => [r.line, seconds(r.mae), seconds(r.median), r.n.toLocaleString('en-US')])} />
    </section>
  )
}

function Importance({ period }: { period: Period }) {
  const rows = period.importance!.filter((row) => row.value > 0)
  return (
    <section className="panel-section">
      <h2>What the model relies on</h2>
      <p className="hint">How much worse the error gets, in seconds, when one input is scrambled.</p>
      <BarList bars={rows.map((row) => ({ color: SERIES.model, key: row.feature, label: row.label, value: row.value }))} format={seconds} />
    </section>
  )
}

function EarlyWarning({ period }: { period: Period }) {
  const rows = period.warning!.filter((row) => row.method === 'model' && row.subset === 'onsets')
  if (!rows.length) return null
  return (
    <section className="panel-section">
      <h2>Warning of a sudden 10+ minute delay</h2>
      <p className="hint">Of trains under 5 minutes late now that end up 10+ minutes late, the share flagged in advance when half the alarms are right.</p>
      <BarList
        bars={rows.map((row) => ({ color: SERIES.model, key: String(row.stops), label: `${row.stops} stop${row.stops > 1 ? 's' : ''} ahead`, note: `${row.positives.toLocaleString('en-US')} cases`, value: row.recall ?? 0 }))}
        format={formatPercent}
        max={1}
      />
    </section>
  )
}
