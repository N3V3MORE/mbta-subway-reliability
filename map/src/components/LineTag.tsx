import { lineName } from '../data/labels'
import type { Line } from '../types'

/** A line's name as riders say it, with its colour beside it. */
export function LineTag({ id, line }: { id: string; line?: Line }) {
  return (
    <span className="line-tag">
      <i style={{ background: line?.color ?? 'var(--muted)' }} />
      {lineName(line, id)}
    </span>
  )
}

/** Compact dots for a station's lines, one per trunk colour. */
export function LineDots({ ids, lineById }: { ids: string[]; lineById: Map<string, Line> }) {
  const colors = [...new Set(ids.map((id) => lineById.get(id)?.color).filter(Boolean))] as string[]
  return <span aria-hidden="true" className="line-dots">{colors.map((color) => <i key={color} style={{ background: color }} />)}</span>
}
