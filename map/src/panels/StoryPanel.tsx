import { useEffect, useRef } from 'react'
import { ArrowRight } from 'lucide-react'
import type { PreparedNetwork } from '../data/network'
import type { Results } from '../types'
import { chapters, type StoryView } from './storyChapters'

export function StoryPanel({ network, onView, results }: { network: PreparedNetwork; onView: (view: StoryView) => void; results: Results }) {
  const list = chapters(network, results)
  const refs = useRef<(HTMLElement | null)[]>([])
  const onViewRef = useRef(onView)
  onViewRef.current = onView

  // The chapter crossing a band near the top of the panel is the one being read.
  useEffect(() => {
    const root = refs.current[0]?.closest('.panel') ?? null
    const observer = new IntersectionObserver((entries) => {
      const seen = entries.filter((e) => e.isIntersecting).map((e) => Number((e.target as HTMLElement).dataset.chapter))
      if (seen.length) onViewRef.current(list[Math.min(...seen)].view)
    }, { root, rootMargin: '-18% 0px -77% 0px' })
    refs.current.forEach((el) => el && observer.observe(el))
    return () => observer.disconnect()
    // The chapter list depends only on the data, which does not change while the panel is open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list.length])

  return (
    <div className="story">
      {list.map((chapter, i) => (
        <section className="chapter" data-chapter={i} key={chapter.question} ref={(el) => { refs.current[i] = el }}>
          {i === 0 ? <h1>{chapter.question}</h1> : <h2>{chapter.question}</h2>}
          <p className="chapter-answer">{chapter.answer}</p>
          {chapter.body ? <div className="chapter-body">{chapter.body}</div> : null}
          <a className="chapter-link" href={chapter.explore.href}>{chapter.explore.label}<ArrowRight size={15} /></a>
        </section>
      ))}
      <p className="story-end">Every number here comes from the project’s pipeline; the method, and everything left out, is in the README.</p>
    </div>
  )
}
