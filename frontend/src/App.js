import React, { useState, useEffect, useRef } from 'react';
import { Room, RoomEvent, Track } from 'livekit-client';
import './App.css';

const LIVEKIT_URL = process.env.REACT_APP_LIVEKIT_URL || 'ws://localhost:7880';
const LIVEKIT_API_KEY = process.env.LIVEKIT_API_KEY || '';
const LIVEKIT_API_SECRET = process.env.LIVEKIT_API_SECRET || '';

function App() {
  const [room, setRoom] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [error, setError] = useState(null);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const transcriptRef = useRef(null);
  const audioElRef = useRef(null);

  // Auto-scroll transcript to bottom
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  const generateAccessToken = async () => {
    try {
      // In a real app, this should be done on your backend
      // For now, we'll use a simple room name and participant identity
      const response = await fetch('http://localhost:8000/api/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          room: 'recipe-agent-room',
          identity: `user-${Date.now()}`,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get access token');
      }

      const data = await response.json();
      return data.token;
    } catch (err) {
        // Remove the fallback call
        // console.warn('Using client-side token generation (not secure for production)');
        // return generateClientSideToken();
        
        // Just throw the error directly
        throw new Error(`Failed to get access token: ${err.message}`);
    }
    
  };

  const startCall = async () => {
    try {
      setIsConnecting(true);
      setError(null);

      // Create room instance
      const newRoom = new Room({
        adaptiveStream: true,
        dynacast: true,
      });

      // Set up event listeners
      newRoom.on(RoomEvent.Connected, () => {
        console.log('Connected to room');
        setIsConnected(true);
        setIsConnecting(false);
      });

      newRoom.on(RoomEvent.Disconnected, () => {
        console.log('Disconnected from room');
        setIsConnected(false);
        setRoom(null);
      });

      newRoom.on(RoomEvent.ParticipantConnected, (participant) => {
        console.log('Participant connected:', participant.identity);
        setupParticipantEvents(participant);
      });

      newRoom.on(RoomEvent.LocalTrackPublished, (publication) => {
        console.log('Local track published:', publication.trackName);
      });

      // Render remote audio
      newRoom.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
        console.log('Track subscribed:', publication.trackName, track.kind, 'from', participant.identity);
        if (track.kind === Track.Kind.Audio && audioElRef.current) {
          try {
            track.attach(audioElRef.current);
          } catch (e) {
            console.warn('Failed to attach audio track:', e);
          }
        }
      });

      // Get access token
      const token = await generateAccessToken();
      if (!token) {
        throw new Error('Failed to generate access token. Please check your configuration.');
      }

      // Connect to room
      await newRoom.connect(LIVEKIT_URL, token);
      // Unlock audio playback (must be called in a user gesture)
      try {
        await newRoom.startAudio();
      } catch (e) {
        console.warn('startAudio failed; user interaction required?', e);
      }
      
      // Enable microphone only (no camera needed)
      await newRoom.localParticipant.setMicrophoneEnabled(true);
      
      setRoom(newRoom);
    } catch (err) {
      console.error('Failed to start call:', err);
      setError(err.message);
      setIsConnecting(false);
    }
  };

  const setupParticipantEvents = (participant) => {
    // Listen for transcript updates
    participant.on('trackSubscribed', (track, publication) => {
      if (publication.trackName === 'transcript') {
        track.on('message', (data) => {
          try {
            const transcriptData = JSON.parse(data);
            addToTranscript(transcriptData);
          } catch (err) {
            console.error('Failed to parse transcript data:', err);
          }
        });
      }
    });
  };

  const addToTranscript = (data) => {
    const timestamp = new Date().toLocaleTimeString();
    setTranscript(prev => [...prev, {
      id: Date.now(),
      timestamp,
      speaker: data.speaker || 'Unknown',
      text: data.text || data.message || '',
      type: data.type || 'speech'
    }]);
  };

  const endCall = async () => {
    if (room) {
      await room.disconnect();
      setRoom(null);
      setIsConnected(false);
      setTranscript([]);
    }
  };

  const toggleMicrophone = async () => {
    if (room && room.localParticipant) {
      if (audioEnabled) {
        await room.localParticipant.setMicrophoneEnabled(false);
      } else {
        await room.localParticipant.setMicrophoneEnabled(true);
      }
      setAudioEnabled(!audioEnabled);
    }
  };

  const clearTranscript = () => {
    setTranscript([]);
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🍳 Chef Ramsay Recipe Agent</h1>
          <p>Your demanding AI cooking mentor powered by voice</p>
        </header>

        <div className="controls">
          {!isConnected ? (
            <button 
              className="btn btn-primary btn-large"
              onClick={startCall}
              disabled={isConnecting}
            >
              {isConnecting ? 'Connecting...' : '🎤 Start Call'}
            </button>
          ) : (
            <div className="call-controls">
              <button 
                className="btn btn-danger btn-large"
                onClick={endCall}
              >
                📞 End Call
              </button>
              <button 
                className={`btn ${audioEnabled ? 'btn-warning' : 'btn-success'}`}
                onClick={toggleMicrophone}
              >
                {audioEnabled ? '🔇 Mute' : '🎤 Unmute'}
              </button>
              <button 
                className="btn btn-secondary"
                onClick={clearTranscript}
              >
                🗑️ Clear
              </button>
            </div>
          )}
        </div>

        {/* Hidden audio element for remote audio playback */}
        <audio ref={audioElRef} autoPlay playsInline />

        {error && (
          <div className="error">
            <p>❌ Error: {error}</p>
            <small>Make sure your LiveKit server is running and configuration is correct.</small>
          </div>
        )}

        <div className="status">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
          </div>
          {isConnected && (
            <div className="audio-status">
              {audioEnabled ? '🎤 Microphone On' : '🔇 Microphone Off'}
            </div>
          )}
        </div>

        <div className="transcript-container">
          <div className="transcript-header">
            <h3>📝 Live Transcript</h3>
            <span className="transcript-count">{transcript.length} messages</span>
          </div>
          
          <div className="transcript" ref={transcriptRef}>
            {transcript.length === 0 ? (
              <div className="transcript-empty">
                <p>🎙️ Start speaking to see the live transcript...</p>
                <p>Ask Chef Ramsay about recipes, meal prep, or cooking techniques!</p>
              </div>
            ) : (
              transcript.map((entry) => (
                <div key={entry.id} className={`transcript-entry ${entry.speaker.toLowerCase()}`}>
                  <div className="transcript-meta">
                    <span className="speaker">{entry.speaker}</span>
                    <span className="timestamp">{entry.timestamp}</span>
                  </div>
                  <div className="transcript-text">{entry.text}</div>
                </div>
              ))
            )}
          </div>
        </div>

        <footer className="footer">
          <p>Powered by LiveKit • Built with React</p>
          <p>Ask about recipes, ingredient substitutions, or cooking techniques!</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
