/** Service-day seconds as a clock time. Times after midnight wrap to 00:xx. */
export function formatClock(seconds: number) {
  const minutes = Math.floor(Math.max(0, seconds) / 60)
  return `${String(Math.floor(minutes / 60) % 24).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`
}

/** A signed delay: "+3:05" late, "−0:20" early. */
export function formatDelay(seconds: number) {
  const total = Math.round(Math.abs(seconds))
  return `${seconds < 0 ? '−' : '+'}${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export const formatPercent = (share: number) => `${Math.round(share * 100)}%`

const DATE_FORMATS = {
  long: new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'long', timeZone: 'UTC', weekday: 'long', year: 'numeric' }),
  short: new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC', year: 'numeric' }),
}

/** "Wednesday 10 June 2026", or "10 Jun 2026" when short. */
export function formatDate(date: string, style: keyof typeof DATE_FORMATS = 'long') {
  return DATE_FORMATS[style].format(new Date(`${date}T00:00:00Z`))
}
