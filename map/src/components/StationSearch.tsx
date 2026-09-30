import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { Search, X } from 'lucide-react'
import { searchStations } from '../data/search'
import type { Line, Station } from '../types'

export function StationSearch({ lineById, onSelect, stations }: {
  lineById: Map<string, Line>
  onSelect: (stationId: string) => void
  stations: Station[]
}) {
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [activeStationId, setActiveStationId] = useState<string>()
  const inputRef = useRef<HTMLInputElement>(null)
  const activeOptionRef = useRef<HTMLButtonElement>(null)
  const listId = useId()
  const matches = useMemo(() => searchStations(stations, query), [stations, query])
  const results = matches.slice(0, 8)
  const activeIndex = Math.max(0, results.findIndex((station) => station.id === activeStationId))
  const activeStation = results[activeIndex]
  const expanded = isOpen && query.trim().length > 0
  const activeOptionId = activeStation ? `${listId}-${activeStation.id}` : undefined

  useEffect(() => {
    if (expanded) activeOptionRef.current?.scrollIntoView({ block: 'nearest' })
  }, [activeOptionId, expanded])

  const selectStation = (station: Station, dismissKeyboard = false) => {
    setQuery(station.name)
    setIsOpen(false)
    onSelect(station.id)
    if (dismissKeyboard) inputRef.current?.blur()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing) return
    if (event.key === 'Escape' || event.key === 'Tab') {
      setIsOpen(false)
      return
    }
    if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && results.length) {
      event.preventDefault()
      const nextIndex = !expanded
        ? event.key === 'ArrowDown' ? 0 : results.length - 1
        : Math.max(0, Math.min(results.length - 1, activeIndex + (event.key === 'ArrowDown' ? 1 : -1)))
      setActiveStationId(results[nextIndex].id)
      setIsOpen(true)
    }
    if (event.key === 'Enter' && expanded && activeStation) {
      event.preventDefault()
      selectStation(activeStation)
    }
  }

  return (
    <section className="station-search" role="search" aria-label="Find a station" onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) setIsOpen(false)
    }}>
      <div className="station-search-field">
        <Search aria-hidden="true" size={17} />
        <input
          aria-activedescendant={expanded ? activeOptionId : undefined}
          aria-autocomplete="list"
          aria-controls={expanded ? listId : undefined}
          aria-expanded={expanded}
          aria-label="Search stations"
          autoComplete="off"
          onChange={(event) => { setQuery(event.target.value); setActiveStationId(undefined); setIsOpen(true) }}
          onClick={() => setIsOpen(true)}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder="Find a station…"
          ref={inputRef}
          role="combobox"
          spellCheck={false}
          type="text"
          value={query}
        />
        {query ? <button aria-label="Clear station search" onClick={() => {
          setQuery('')
          setActiveStationId(undefined)
          inputRef.current?.focus()
        }} type="button"><X aria-hidden="true" size={16} /></button> : null}
      </div>
      {expanded ? (
        <div className="station-search-popup">
          <ul aria-label="Matching stations" id={listId} role="listbox">
            {results.map((station, index) => (
              <li key={station.id} role="none">
                <button
                  aria-selected={index === activeIndex}
                  className="station-search-option"
                  id={`${listId}-${station.id}`}
                  onClick={() => selectStation(station, true)}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setActiveStationId(station.id)}
                  ref={index === activeIndex ? activeOptionRef : undefined}
                  role="option"
                  tabIndex={-1}
                  type="button"
                >
                  <span className="station-search-name">{station.name}</span>
                  <span className="station-search-lines">
                    {station.lines.map((lineId) => {
                      const line = lineById.get(lineId)
                      return line ? <span key={lineId}><i aria-hidden="true" style={{ backgroundColor: line.color }} />{line.short}</span> : null
                    })}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <p role="status">{matches.length === 0 ? 'No stations match.'
            : matches.length > results.length ? `Showing ${results.length} of ${matches.length} stations. Refine your search.`
              : `${matches.length} ${matches.length === 1 ? 'station' : 'stations'} · ↑↓ navigate · Enter select · Esc close`}</p>
        </div>
      ) : null}
    </section>
  )
}
