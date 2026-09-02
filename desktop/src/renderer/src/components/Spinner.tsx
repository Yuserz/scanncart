import { type JSX } from 'react'
import './Spinner.css'

// Decorative loading ring; the accompanying text carries the meaning, so
// this is aria-hidden rather than a live region.
export function Spinner({ size = 16 }: { size?: number }): JSX.Element {
  return <span className="spinner" aria-hidden="true" style={{ width: size, height: size }} />
}
