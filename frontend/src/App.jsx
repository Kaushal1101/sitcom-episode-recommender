import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const MULTI_TYPES = new Set(['ATTRIBUTE_MULTI', 'SOFT_CRITIQUE'])
const YN_TYPES = new Set(['ATTRIBUTE_YN', 'ITEM_YN', 'RECOMMEND'])

function PreferencesPanel({ confidence, preferences }) {
  if (preferences == null) return null

  const moodEntries = [
    ['humor', preferences.mood?.humor],
    ['energy', preferences.mood?.energy],
    ['comfort', preferences.mood?.comfort],
    ['sadness', preferences.mood?.sadness],
  ]

  const toneEntries = Object.entries(preferences.tone ?? {})
    .filter(([, value]) => Math.abs(value) > 0.01)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))

  return (
    <details>
      <summary>Debug: Confidence &amp; Preferences</summary>
      {confidence != null && (
        <div className="pref-section">
          <h4>Confidence: {confidence.toFixed(3)}</h4>
          <div className="confidence-bar-track">
            <div
              className="confidence-bar"
              style={{ width: `${(confidence * 100).toFixed(0)}%` }}
            />
          </div>
        </div>
      )}
      <div className="pref-section">
        <h4>Mood</h4>
        {moodEntries.map(([label, value]) => (
          <div key={label} className="pref-row">
            <span>{label}</span>
            <span>{(value ?? 0).toFixed(3)}</span>
          </div>
        ))}
      </div>
      <div className="pref-section">
        <h4>Tone</h4>
        {toneEntries.map(([label, value]) => (
          <div key={label} className="pref-row">
            <span>{label}</span>
            <span>{value.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </details>
  )
}

function App() {
  const [screen, setScreen] = useState('start')
  const [sessionId, setSessionId] = useState(null)
  const [question, setQuestion] = useState('')
  const [interactionType, setInteractionType] = useState('')
  const [options, setOptions] = useState(null)
  const [candidateSynopses, setCandidateSynopses] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [preferences, setPreferences] = useState(null)
  const [farewell, setFarewell] = useState(null)
  const [loading, setLoading] = useState(false)
  const [inputValue, setInputValue] = useState('')

  const resetState = () => {
    setSessionId(null)
    setQuestion('')
    setInteractionType('')
    setOptions(null)
    setCandidateSynopses(null)
    setConfidence(null)
    setPreferences(null)
    setFarewell(null)
    setLoading(false)
    setInputValue('')
  }

  const startSession = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/session`, { method: 'POST' })
      if (!response.ok) {
        setLoading(false)
        return
      }
      const data = await response.json()
      setSessionId(data.session_id)
      setQuestion(data.question)
      setInteractionType(data.interaction_type)
      setOptions(data.options)
      setCandidateSynopses(data.candidate_synopses ?? null)
      setConfidence(data.confidence ?? null)
      setPreferences(data.preferences ?? null)
      setInputValue('')
      setScreen('conversation')
    } finally {
      setLoading(false)
    }
  }

  const handleAnswer = async (answer) => {
    setLoading(true)
    try {
      const response = await fetch(
        `${API_URL}/session/${sessionId}/answer`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ answer }),
        },
      )
      if (!response.ok) {
        setLoading(false)
        return
      }
      const data = await response.json()
      if (data.done) {
        setFarewell(data.farewell ?? 'Goodbye!')
        setScreen('done')
        fetch(`${API_URL}/session/${sessionId}`, { method: 'DELETE' })
      } else {
        setQuestion(data.question)
        setInteractionType(data.interaction_type)
        setOptions(data.options)
        setCandidateSynopses(data.candidate_synopses ?? null)
        setConfidence(data.confidence ?? null)
        setPreferences(data.preferences ?? null)
        setInputValue('')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleStartOver = () => {
    resetState()
    setScreen('start')
  }

  if (screen === 'start') {
    return (
      <>
        <h1>Sitcom Episode Recommender</h1>
        <div className="submit">
          <button type="button" onClick={startSession} disabled={loading}>
            Start
          </button>
        </div>
      </>
    )
  }

  if (screen === 'done') {
    return (
      <>
        <h1>Sitcom Episode Recommender</h1>
        <p className="farewell">{farewell}</p>
        <div className="submit">
          <button type="button" onClick={handleStartOver}>
            Start over
          </button>
        </div>
      </>
    )
  }

  return (
    <>
      <h1>Sitcom Episode Recommender</h1>
      <p className="question">{question}</p>

      {interactionType === 'OPEN_CHAT' && (
        <div className="text-input">
          <input
            type="text"
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            disabled={loading}
          />
          <button
            type="button"
            onClick={() => {
              const trimmed = inputValue.trim()
              if (trimmed) handleAnswer(trimmed)
            }}
            disabled={loading || !inputValue.trim()}
          >
            Submit
          </button>
        </div>
      )}

      {MULTI_TYPES.has(interactionType) && (
        <div className="options">
          {(options ?? []).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => handleAnswer(option)}
              disabled={loading}
            >
              {option}
            </button>
          ))}
        </div>
      )}

      {interactionType === 'ITEM_MULTI' && (
        <div className="options">
          {(options ?? []).map((option, index) => (
            <div key={option} className="episode-option">
              <button
                type="button"
                onClick={() => handleAnswer(option)}
                disabled={loading}
              >
                {option}
              </button>
              <p className="synopsis">{candidateSynopses?.[index] ?? ''}</p>
            </div>
          ))}
        </div>
      )}

      {YN_TYPES.has(interactionType) && (
        <>
          {interactionType === 'ITEM_YN' && (
            <p className="synopsis">{candidateSynopses?.[0] ?? ''}</p>
          )}
          <div className="yn">
            <button
              type="button"
              onClick={() => handleAnswer('Yes')}
              disabled={loading}
            >
              Yes
            </button>
            <button
              type="button"
              onClick={() => handleAnswer('No')}
              disabled={loading}
            >
              No
            </button>
          </div>
        </>
      )}

      {interactionType === 'RATING' && (
        <div className="rating">
          {['1', '2', '3', '4', '5'].map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => handleAnswer(value)}
              disabled={loading}
            >
              {value}
            </button>
          ))}
        </div>
      )}

      <PreferencesPanel confidence={confidence} preferences={preferences} />
    </>
  )
}

export default App
