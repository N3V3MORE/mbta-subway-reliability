import { describe, expect, it } from 'vitest'
import { chapters } from './storyChapters'
import type { PreparedNetwork } from '../data/network'
import type { Period, Results } from '../types'

const network = {
  lineById: new Map(), lines: [], patterns: [], stationById: new Map(), stations: [],
  replays: [{ date: '2026-06-10' }, { date: '2026-02-23' }],
  window: ['2026-04-02', '2026-06-30'],
} as unknown as PreparedNetwork

const regression = [
  { id: 'hist_gradient_boosting_change', mae: 16.2 }, { id: 'baseline_persistence', mae: 49 }, { id: 'baseline_run_time', mae: 14.9 },
]
const spring = {
  ablation: [{ features: 'schedule', mae: 250 }, { features: '+ delay propagation', mae: 16 }, { features: '+ weather', mae: 16 }],
  arrivals: 3_000_000,
  bandsByLine: [{ line: 'Orange', shares: [0.8, 0.16, 0.03, 0.01] }, { line: 'Green-D', shares: [0.6, 0.24, 0.1, 0.06] }],
  horizons: [{ model: 67.9, n: 1, ownTrain: 68, persistence: 103, runTime: 79.1, stops: 5 }],
  id: 'spring', label: 'Spring 2026', lateShare: 0.1, regression,
  warning: [{ method: 'model', n: 1, positives: 1, prAuc: 0.3, recall: 0.26, stops: 1, subset: 'onsets' }],
} as unknown as Period
const winter = { id: 'winter', label: 'Winter', regression } as unknown as Period

describe('story chapters', () => {
  it('reads in order, each with the map view it shows', () => {
    const list = chapters(network, { crossSeason: null, periods: [spring, winter] } as Results)
    expect(list.map((c) => c.question)).toEqual([
      'The T, as it actually ran', 'How late do trains run?', 'Can we predict how late the next train will be?',
      'What doesn’t help?', 'What can’t be predicted?', 'Does it hold up?',
    ])
    expect(list.map((c) => c.view)).toEqual([
      { metric: 'entries', mode: 'network' },
      { metric: 'late', mode: 'network' },
      { mode: 'replay', replayDate: '2026-06-10', seconds: 28_800 },
      { metric: 'late', mode: 'network' },
      { mode: 'replay', replayDate: '2026-02-23', seconds: 28_800 },
      { metric: 'late', mode: 'network' },
    ])
  })

  it('leaves out a chapter whose data was not exported', () => {
    const bare = { ...spring, bandsByLine: undefined, warning: undefined } as Period
    const questions = chapters(network, { crossSeason: null, periods: [bare] } as Results).map((c) => c.question)
    expect(questions).not.toContain('How late do trains run?')
    expect(questions).not.toContain('What can’t be predicted?')
    expect(questions).not.toContain('Does it hold up?')
  })

  it('has nothing to tell without results', () => {
    expect(chapters(network, { crossSeason: null, periods: [] })).toEqual([])
  })
})
