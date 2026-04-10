import { useState, useRef, useEffect } from 'react'
import imageCompression from 'browser-image-compression'
import { chatWithRAG } from '../api/client'

export default function FloatingChat() {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      text: '♻️ Hi! I\'m EcoStream AI Assistant. Ask me anything about waste disposal, recycling rules, or waste management in your city. You can also attach a photo of waste!',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [attachedFile, setAttachedFile] = useState(null)
  const [attachedPreview, setAttachedPreview] = useState(null)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function handleAttachFile(e) {
    const file = e.target.files?.[0]
    if (file) {
      setAttachedFile(file)
      const reader = new FileReader()
      reader.onload = () => setAttachedPreview(reader.result)
      reader.readAsDataURL(file)
    }
  }

  function removeAttachment() {
    setAttachedFile(null)
    setAttachedPreview(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    if (cameraInputRef.current) cameraInputRef.current.value = ''
  }

  async function handleSend() {
    const trimmed = input.trim()
    if (!trimmed && !attachedFile) return

    const userMsg = {
      role: 'user',
      text: trimmed || '📷 [Photo attached]',
      image: attachedPreview,
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      let fileToSend = null
      if (attachedFile) {
        fileToSend = await imageCompression(attachedFile, {
          maxSizeMB: 1.0,
          maxWidthOrHeight: 1024,
        })
      }

      removeAttachment()

      const res = await chatWithRAG(
        trimmed || 'What waste materials are in this image? How should I dispose of them?',
        'Bangalore',
        fileToSend
      )

      let botText = res.answer || 'Sorry, I couldn\'t find an answer.'
      if (res.detected_materials?.length > 0) {
        botText = `🔍 **Detected:** ${res.detected_materials.join(', ')}\n\n${botText}`
      }

      setMessages((prev) => [...prev, { role: 'bot', text: botText }])
    } catch (err) {
      console.error('Chat error:', err)
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: '⚠️ Something went wrong. Please try again.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <>
      {/* Backdrop blur when chat is open */}
      {isOpen && (
        <div className="fc-backdrop" onClick={() => setIsOpen(false)} />
      )}

      {/* Floating button */}
      <button
        className="fc-fab"
        onClick={() => setIsOpen(!isOpen)}
        title="Chat with EcoStream AI"
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Chat popup */}
      {isOpen && (
        <div className="fc-popup">
          {/* Header */}
          <div className="fc-header">
            <div className="fc-header__info">
              <span className="fc-header__avatar">♻️</span>
              <div>
                <div className="fc-header__title">EcoStream AI</div>
                <div className="fc-header__subtitle">Waste disposal assistant</div>
              </div>
            </div>
            <button className="fc-header__close" onClick={() => setIsOpen(false)}>✕</button>
          </div>

          {/* Messages */}
          <div className="fc-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`fc-msg fc-msg--${msg.role}`}>
                {msg.image && (
                  <img src={msg.image} alt="Attached" className="fc-msg__image" />
                )}
                <div className="fc-msg__bubble">
                  {msg.text.split('\n').map((line, j) => (
                    <span key={j}>
                      {line.replace(/\*\*(.*?)\*\*/g, '$1')}
                      {j < msg.text.split('\n').length - 1 && <br />}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            {loading && (
              <div className="fc-msg fc-msg--bot">
                <div className="fc-msg__bubble fc-msg__typing">
                  <span className="fc-dot" /><span className="fc-dot" /><span className="fc-dot" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Attachment preview */}
          {attachedPreview && (
            <div className="fc-attachment">
              <img src={attachedPreview} alt="Preview" className="fc-attachment__img" />
              <button className="fc-attachment__remove" onClick={removeAttachment}>✕</button>
            </div>
          )}

          {/* Input bar */}
          <div className="fc-input-bar">
            {/* File attach */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={handleAttachFile}
            />
            <button
              className="fc-input-btn"
              onClick={() => fileInputRef.current?.click()}
              title="Attach photo"
            >
              📎
            </button>

            {/* Camera capture */}
            <input
              ref={cameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              style={{ display: 'none' }}
              onChange={handleAttachFile}
            />
            <button
              className="fc-input-btn"
              onClick={() => cameraInputRef.current?.click()}
              title="Take photo"
            >
              📸
            </button>

            {/* Text input */}
            <input
              className="fc-input-text"
              type="text"
              placeholder="Ask about waste disposal..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />

            {/* Send */}
            <button
              className="fc-input-btn fc-input-btn--send"
              onClick={handleSend}
              disabled={loading || (!input.trim() && !attachedFile)}
              title="Send"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  )
}
