import type { Line } from '../types'

/** "Red", "Green E", "Mattapan": a line's name as riders say it. */
export function lineName(line: Line | undefined, id: string) {
  if (!line) return id
  return line.id.startsWith('Green-') ? `Green ${line.short}` : line.short
}

/**
 * Chart series with a fixed identity across every chart: our model, the "stays as
 * late as it is" rule of thumb, and the running-time lookup. Validated together
 * on the panel surface; the values live in styles.css.
 */
export const SERIES = {
  lookup: 'var(--series-lookup)',
  model: 'var(--series-model)',
  persistence: 'var(--series-persistence)',
}

/** Cluster names as written by Track B ("less reliable", "morning-peaked"), in sentence case for display. */
export function clusterLabel(name: string | null) {
  if (!name) return null
  const text = name.replace(/-peaked$/, ' peak')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/** "16 s", or "9.4 s" below ten. */
export const seconds = (value: number) => `${value < 10 ? value.toFixed(1) : Math.round(value)} s`
