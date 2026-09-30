import { describe, expect, it } from 'vitest'
import { clusterLabel, lineName, seconds } from './labels'

describe('labels', () => {
  it('names lines the way riders say them', () => {
    expect(lineName({ color: '#00843D', id: 'Green-E', name: 'Green Line E', short: 'E' }, 'Green-E')).toBe('Green E')
    expect(lineName({ color: '#DA291C', id: 'Red', name: 'Red Line', short: 'Red' }, 'Red')).toBe('Red')
    expect(lineName(undefined, 'Silver')).toBe('Silver')
  })

  it('writes cluster names in sentence case', () => {
    expect(clusterLabel('less reliable')).toBe('Less reliable')
    expect(clusterLabel('morning-peaked')).toBe('Morning peak')
    expect(clusterLabel(null)).toBeNull()
  })

  it('keeps a decimal only below ten seconds', () => {
    expect(seconds(9.44)).toBe('9.4 s')
    expect(seconds(16.21)).toBe('16 s')
  })
})
