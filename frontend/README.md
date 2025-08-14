# Chef Ramsay Recipe Agent Frontend

A React-based frontend for the Chef Ramsay voice AI agent, built with LiveKit for real-time voice communication and live transcription.

## Features

- 🎤 **Start/End Call**: Connect to the Chef Ramsay voice agent
- 📝 **Live Transcript**: Real-time display of conversation with the agent
- 🔇 **Microphone Control**: Mute/unmute during the call
- 🎨 **Modern UI**: Beautiful, responsive design with gradient backgrounds
- 📱 **Mobile Friendly**: Works on desktop and mobile devices

## Quick Start

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your LiveKit credentials
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Open your browser**:
   Navigate to `http://localhost:3000`

## Configuration

Create a `.env` file in the frontend directory with your LiveKit configuration:

```env
REACT_APP_LIVEKIT_URL=ws://localhost:7880
REACT_APP_LIVEKIT_API_KEY=your_api_key_here
REACT_APP_LIVEKIT_API_SECRET=your_api_secret_here
```

For production, use your LiveKit Cloud URL:
```env
REACT_APP_LIVEKIT_URL=wss://your-project.livekit.cloud
```

## How to Use

1. **Start the Agent**: Make sure your Python agent is running (`python agent.py`)
2. **Open Frontend**: Launch the React app with `npm start`
3. **Start Call**: Click the "🎤 Start Call" button
4. **Talk to Chef Ramsay**: Ask about recipes, meal prep, or cooking techniques
5. **View Transcript**: Watch the live conversation transcript update in real-time
6. **End Call**: Click "📞 End Call" when finished

## Components

### Main Features
- **Connection Management**: Handles LiveKit room connection/disconnection
- **Audio Controls**: Microphone mute/unmute functionality
- **Live Transcript**: Real-time display of speech-to-text
- **Error Handling**: User-friendly error messages
- **Responsive Design**: Works on all screen sizes

### Transcript Display
- Color-coded messages by speaker (User vs Agent)
- Timestamps for each message
- Auto-scroll to latest messages
- Message counter
- Clear transcript option

## Development

### Available Scripts

- `npm start`: Start development server
- `npm build`: Build for production
- `npm test`: Run tests
- `npm eject`: Eject from Create React App (not recommended)

### Dependencies

- **React 18**: Modern React with hooks
- **LiveKit Client**: Real-time communication
- **LiveKit React Components**: Pre-built UI components
- **Create React App**: Development tooling

## Troubleshooting

### Common Issues

1. **Token Generation Error**: 
   - Make sure your LiveKit credentials are correct
   - For production, implement server-side token generation

2. **Connection Failed**:
   - Verify your LiveKit server is running
   - Check the WebSocket URL is correct
   - Ensure firewall allows WebSocket connections

3. **No Audio**:
   - Check browser microphone permissions
   - Verify microphone is not muted
   - Try refreshing the page

4. **Transcript Not Updating**:
   - Ensure the agent is sending transcript events
   - Check browser console for errors
   - Verify WebSocket connection is active

## Security Note

The current implementation includes client-side token generation for development purposes. In production, you should:

1. Generate tokens on your backend server
2. Never expose API keys in client-side code
3. Implement proper authentication and authorization
4. Use HTTPS/WSS for all connections

## Next Steps

- Implement server-side token generation
- Add user authentication
- Enhanced error handling and retry logic
- Voice activity indicators
- Recording and playback features
- Mobile app version with React Native
