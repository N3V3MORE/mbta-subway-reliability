import { type ReactNode, useLayoutEffect, useRef, useState } from 'react'
import { SERIES } from '../data/labels'

/**
 * Small charts for the results panel, drawn to one spec: hairline grid, 2px lines,
 * bars with a 4px rounded end, values in text colours, a legend for two or more
 * series, a hover readout, and a table view behind every chart. Series colours
 * come from `SERIES` (data/labels.ts).
 */

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return undefined
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

/** A "show the numbers" disclosure: every chart's values, reachable without hovering. */
export function TableView({ columns, rows }: { columns: string[]; rows: ReactNode[][] }) {
  return (
    <details className="table-view">
      <summary>Show the numbers</summary>
      <div className="table-scroll">
        <table>
          <thead><tr>{columns.map((column) => <th key={column} scope="col">{column}</th>)}</tr></thead>
          <tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </details>
  )
}

export function Legend({ items }: { items: { color: string; label: string; shape?: 'line' | 'box' }[] }) {
  return (
    <ul className="chart-legend">
      {items.map((item) => (
        <li key={item.label}><i className={item.shape === 'line' ? 'key-line' : 'key-box'} style={{ background: item.color }} />{item.label}</li>
      ))}
    </ul>
  )
}

export type Bar = { color?: string; key: string; label: ReactNode; note?: string; value: number }

/** Horizontal bars from one baseline, value at the tip. One series, so no legend. */
export function BarList({ bars, format, max }: { bars: Bar[]; format: (value: number) => string; max?: number }) {
  const top = max ?? Math.max(...bars.map((bar) => bar.value))
  return (
    <ul className="bar-list">
      {bars.map((bar) => (
        <li key={bar.key} title={`${typeof bar.label === 'string' ? bar.label : bar.key}: ${format(bar.value)}`}>
          <span className="bar-label">{bar.label}{bar.note ? <small>{bar.note}</small> : null}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ background: bar.color ?? 'var(--series-context)', width: `${Math.max(0.5, (bar.value / top) * 100)}%` }} />
            <span className="bar-value">{format(bar.value)}</span>
          </span>
        </li>
      ))}
    </ul>
  )
}

export type Series = { color: string; key: string; label: string; values: (number | null)[] }

/**
 * Lines over a shared x axis, with a crosshair that snaps to the nearest x and
 * one readout listing every series there. Direct labels at the right end.
 */
export function LineChart({ format, height = 220, series, x, xFormat, xLabel }: {
  format: (value: number) => string
  height?: number
  series: Series[]
  x: number[]
  xFormat: (value: number) => string
  xLabel: string
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number>()
  const pad = { bottom: 34, left: 44, right: 118, top: 12 }
  const w = Math.max(0, width - pad.left - pad.right)
  const h = height - pad.top - pad.bottom
  const all = series.flatMap((s) => s.values).filter((v): v is number => v !== null)
  const top = niceMax(Math.max(...all))
  const [x0, x1] = [Math.min(...x), Math.max(...x)]
  const px = (value: number) => pad.left + ((value - x0) / (x1 - x0 || 1)) * w
  const py = (value: number) => pad.top + h - (value / top) * h
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * top)
  const ends = placeLabels(series.map((s) => py(lastValue(s.values))), 14)

  const onMove = (event: React.PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    const at = x0 + ((event.clientX - box.left) / box.width) * (x1 - x0)
    setHover(x.reduce((best, value, i) => (Math.abs(value - at) < Math.abs(x[best] - at) ? i : best), 0))
  }

  return (
    <div className="line-chart" ref={ref}>
      {width > 0 ? (
        <svg aria-label={`${series.map((s) => s.label).join(', ')} by ${xLabel}`} height={height} role="img" width={width}>
          {ticks.map((tick) => (
            <g key={tick}>
              <line className="grid" x1={pad.left} x2={pad.left + w} y1={py(tick)} y2={py(tick)} />
              <text className="tick" dominantBaseline="middle" textAnchor="end" x={pad.left - 8} y={py(tick)}>{format(tick)}</text>
            </g>
          ))}
          {x.map((value) => <text className="tick" key={value} textAnchor="middle" x={px(value)} y={pad.top + h + 18}>{xFormat(value)}</text>)}
          <text className="axis-title" textAnchor="middle" x={pad.left + w / 2} y={height - 2}>{xLabel}</text>
          {hover !== undefined ? <line className="crosshair" x1={px(x[hover])} x2={px(x[hover])} y1={pad.top} y2={pad.top + h} /> : null}
          {series.map((s, i) => (
            <g key={s.key}>
              <polyline fill="none" points={s.values.flatMap((v, j) => (v === null ? [] : [`${px(x[j])},${py(v)}`])).join(' ')} stroke={s.color} strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
              {s.values.map((v, j) => (v === null ? null : <circle className="dot" cx={px(x[j])} cy={py(v)} fill={s.color} key={j} r={hover === j ? 5 : 4} />))}
              <text className="end-label" dominantBaseline="middle" x={pad.left + w + 12} y={ends[i]}>{s.label}</text>
            </g>
          ))}
          <rect fill="transparent" height={h} onPointerLeave={() => setHover(undefined)} onPointerMove={onMove} width={w + 16} x={pad.left - 8} y={pad.top} />
        </svg>
      ) : <div style={{ height }} />}
      {hover !== undefined && width > 0 ? (
        <div className="tooltip" style={{ left: Math.min(px(x[hover]) + 12, width - 170), top: pad.top }}>
          <p>{xFormat(x[hover])} · {xLabel}</p>
          {series.map((s) => s.values[hover] === null ? null : (
            <p key={s.key}><i className="key-line" style={{ background: s.color }} /><b>{format(s.values[hover]!)}</b> {s.label}</p>
          ))}
        </div>
      ) : null}
    </div>
  )
}

/** Stacked 100% bars, one per row, segments separated by a 2px surface gap. */
export function StackedBars({ colors, labels, rows, onHover }: {
  colors: readonly string[]
  labels: string[]
  onHover?: (key: string | undefined) => void
  rows: { key: string; label: ReactNode; shares: number[] }[]
}) {
  const [hover, setHover] = useState<{ row: number; segment: number }>()
  return (
    <div className="stacked" onPointerLeave={() => { setHover(undefined); onHover?.(undefined) }}>
      <Legend items={labels.map((label, i) => ({ color: colors[i], label }))} />
      {rows.map((row, r) => (
        <div className="stacked-row" key={row.key} onPointerEnter={() => onHover?.(row.key)}>
          <span className="bar-label">{row.label}</span>
          <span className="stacked-track">
            {row.shares.map((share, s) => share > 0 ? (
              <span
                className={hover?.row === r && hover.segment === s ? 'is-hover' : ''}
                key={s}
                onPointerEnter={() => setHover({ row: r, segment: s })}
                style={{ background: colors[s], flexGrow: share }}
                title={`${labels[s]}: ${(share * 100).toFixed(1)}%`}
              />
            ) : null)}
          </span>
          <span className="stacked-value">{Math.round((row.shares[2] + row.shares[3]) * 100)}%</span>
        </div>
      ))}
    </div>
  )
}

/** A grid of cells shaded on one hue, lighter = more. Blank where the data is too thin. */
export function Heatmap({ cols, colFormat, format, rows }: {
  colFormat: (col: number) => string
  cols: number[]
  format: (value: number) => string
  rows: { key: string; label: ReactNode; values: (number | null)[] }[]
}) {
  const max = Math.max(...rows.flatMap((row) => row.values).filter((v): v is number => v !== null))
  const [hover, setHover] = useState<string>()
  return (
    <div className="heatmap" style={{ '--cols': cols.length } as React.CSSProperties}>
      {rows.map((row) => (
        <div className="heat-row" key={row.key}>
          <span className="bar-label">{row.label}</span>
          {row.values.map((value, i) => (
            <span
              aria-label={value === null ? undefined : `${colFormat(cols[i])}: ${format(value)}`}
              className={`heat-cell${value === null ? ' is-empty' : ''}`}
              key={cols[i]}
              onPointerEnter={() => setHover(value === null ? undefined : `${row.key} · ${colFormat(cols[i])}: ${format(value)}`)}
              style={value === null ? undefined : { background: heat(value / max) }}
            />
          ))}
        </div>
      ))}
      <div className="heat-row heat-axis">
        <span />
        {cols.map((col, i) => <span key={col}>{i % 3 === 0 ? colFormat(col) : ''}</span>)}
      </div>
      <div className="heat-foot">
        <span className="heat-scale"><i style={{ background: heat(0) }} /><i style={{ background: heat(0.5) }} /><i style={{ background: heat(1) }} /></span>
        <span>0 to {format(max)}</span>
        <span className="heat-readout">{hover ?? 'Hover a cell for its value'}</span>
      </div>
    </div>
  )
}

/** Two values per row joined by a rule: where the rule of thumb was, and where the model got to. */
export function Dumbbell({ format, rows }: { format: (value: number) => string; rows: { from: number; key: string; label: ReactNode; to: number }[] }) {
  const top = niceMax(Math.max(...rows.flatMap((row) => [row.from, row.to])))
  const at = (value: number) => `${(value / top) * 100}%`
  return (
    <div className="dumbbell">
      <Legend items={[{ color: SERIES.persistence, label: 'Stays as late as it is' }, { color: SERIES.model, label: 'Our model' }]} />
      {rows.map((row) => (
        <div className="dumbbell-row" key={row.key} title={`Rule of thumb ${format(row.from)}, model ${format(row.to)}`}>
          <span className="bar-label">{row.label}</span>
          <span className="dumbbell-track">
            <span className="dumbbell-rule" style={{ left: at(Math.min(row.from, row.to)), width: `calc(${at(Math.abs(row.from - row.to))})` }} />
            <span className="dumbbell-dot" style={{ background: SERIES.persistence, left: at(row.from) }} />
            <span className="dumbbell-dot" style={{ background: SERIES.model, left: at(row.to) }} />
            <span className="dumbbell-value" style={{ left: at(row.to) }}>{format(row.to)}</span>
            <span className="dumbbell-value is-from" style={{ left: at(row.from) }}>{format(row.from)}</span>
          </span>
        </div>
      ))}
    </div>
  )
}

/** One hue from the panel surface to a light pink: lighter is more. */
function heat(t: number) {
  const stops = [[58, 36, 49], [110, 42, 85], [168, 56, 127], [220, 95, 168], [245, 179, 214]]
  const scaled = Math.min(1, Math.max(0, t)) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(scaled))
  const f = scaled - i
  const [r, g, b] = stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f))
  return `rgb(${r} ${g} ${b})`
}

function niceMax(value: number) {
  const step = 10 ** Math.floor(Math.log10(value || 1))
  return Math.ceil(value / step) * step
}

function lastValue(values: (number | null)[]) {
  for (let i = values.length - 1; i >= 0; i -= 1) if (values[i] !== null) return values[i]!
  return 0
}

/** Spread end labels apart by at least `gap` pixels, keeping their order. */
function placeLabels(ys: number[], gap: number) {
  const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0])
  const placed = new Array<number>(ys.length)
  let last = -Infinity
  for (const [y, i] of order) {
    placed[i] = Math.max(y, last + gap)
    last = placed[i]
  }
  return placed
}
