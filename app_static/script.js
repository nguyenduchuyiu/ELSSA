class ELSSAWebSocketClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isRecording = false;
        this.isWakeWordListening = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.systemState = 'connecting';
        this.currentAssistantMessage = '';
        
        this.initializeElements();
        this.connectWebSocket();
        
        // Start wake word listening immediately
        this.requestMicrophoneAndStartWakeWordListening();
    }

    initializeElements() {
        // Status elements
        this.statusText = document.getElementById('status-text');
        this.statusDot = document.getElementById('status-dot');
        this.connectionStatus = document.getElementById('connection-status');
        
        // Control buttons
        this.wakeBtn = document.getElementById('wake-btn');
        this.sleepBtn = document.getElementById('sleep-btn');
        this.clearLogsBtn = document.getElementById('clear-logs-btn');
        this.clearChatBtn = document.getElementById('clear-chat-btn');
        
        // Display elements
        this.systemLogs = document.getElementById('system-logs');
        this.conversationChat = document.getElementById('conversation-chat');
        this.recordingIndicator = document.getElementById('recording-indicator');
        this.audioLevel = document.getElementById('audio-level');
        
        // Attach event listeners
        this.wakeBtn.addEventListener('click', () => this.wakeSystem());
        this.sleepBtn.addEventListener('click', () => this.sleepSystem());
        this.clearLogsBtn.addEventListener('click', () => this.clearSystemLogs());
        this.clearChatBtn.addEventListener('click', () => this.clearConversationChat());
    }

    async requestMicrophoneAndStartWakeWordListening() {
        try {
            // Clear initial placeholder logs
            this.systemLogs.innerHTML = '';
            
            // Request microphone permission and start listening for wake word
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            this.addSystemLog('🎤 Microphone access granted');
            this.addSystemLog('👂 Listening for wake word...');
            
            // Start wake word listening
            await this.startWakeWordListening(stream);
            
        } catch (error) {
            console.error('Error requesting microphone:', error);
            this.addSystemLog('❌ Microphone access denied. Click "Wake Up" to try again.');
            this.addSystemMessage('🎤 Please allow microphone access to use wake word detection.');
        }
    }

    async startWakeWordListening(stream) {
        if (this.isWakeWordListening) return;
        
        try {
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    // In a real implementation, this would be sent to wake word detection
                    // For now, we simulate wake word detection
                }
            };
            
            this.mediaRecorder.onstop = () => {
                if (this.isWakeWordListening) {
                    // Restart recording for continuous wake word listening
                    setTimeout(() => {
                        if (this.isWakeWordListening && this.systemState === 'idle') {
                            this.mediaRecorder.start(1000);
                        }
                    }, 100);
                }
            };
            
            // Start recording in chunks for wake word detection
            this.mediaRecorder.start(1000);
            this.isWakeWordListening = true;
            
            // Show that we're listening for wake word
            this.updateWakeWordListeningUI(true);
            
            // Setup audio level visualization for wake word listening
            this.setupWakeWordAudioVisualization(stream);
            
        } catch (error) {
            console.error('Error starting wake word listening:', error);
            this.addSystemLog('❌ Error starting wake word detection');
        }
    }

    stopWakeWordListening() {
        if (!this.isWakeWordListening) return;
        
        this.isWakeWordListening = false;
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        this.updateWakeWordListeningUI(false);
        this.addSystemLog('🔇 Wake word listening stopped');
    }

    updateWakeWordListeningUI(isListening) {
        const indicatorSpan = this.recordingIndicator.querySelector('span');
        
        if (this.systemState === 'idle' && isListening) {
            // Show wake word listening indicator
            this.recordingIndicator.classList.remove('hidden');
            indicatorSpan.textContent = 'Listening for wake word...';
            this.audioLevel.classList.add('active');
        } else if (this.systemState === 'active' || this.systemState === 'active_listening') {
            // Show conversation listening indicator
            this.recordingIndicator.classList.remove('hidden');
            indicatorSpan.textContent = 'Listening...';
            this.audioLevel.classList.add('active');
        } else if (this.systemState === 'thinking') {
            // Show thinking indicator
            this.recordingIndicator.classList.remove('hidden');
            indicatorSpan.textContent = 'Thinking...';
            this.audioLevel.classList.remove('active');
        } else if (this.systemState === 'speaking') {
            // Show speaking indicator
            this.recordingIndicator.classList.remove('hidden');
            indicatorSpan.textContent = 'Speaking...';
            this.audioLevel.classList.remove('active');
        } else {
            // Hide indicator for other states
            this.recordingIndicator.classList.add('hidden');
            this.audioLevel.classList.remove('active');
        }
    }

    setupWakeWordAudioVisualization(stream) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const microphone = audioContext.createMediaStreamSource(stream);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        microphone.connect(analyser);
        analyser.fftSize = 256;
        
        const updateAudioLevel = () => {
            if (!this.isWakeWordListening && !this.isRecording) return;
            
            analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            const level = Math.min(100, (average / 255) * 100);
            
            // Update audio level bars
            const bars = this.audioLevel.querySelectorAll('.level-bar');
            bars.forEach((bar, index) => {
                const barLevel = Math.max(10, (level * (index + 1)) / bars.length);
                bar.style.height = `${barLevel}%`;
            });
            
            requestAnimationFrame(updateAudioLevel);
        };
        
        updateAudioLevel();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            this.isConnected = true;
            this.updateConnectionStatus('Connected');
            this.addSystemLog('🔗 Connected to ELSSA');
            console.log('WebSocket connected');
        };
        
        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.websocket.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus('Disconnected');
            this.updateSystemState('connecting');
            this.addSystemLog('🔌 Connection lost. Reconnecting...');
            console.log('WebSocket disconnected');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    this.connectWebSocket();
                }
            }, 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.addSystemLog('❌ Connection error');
        };
    }

    handleWebSocketMessage(message) {
        const { type, data } = message;
        
        switch (type) {
            case 'state_change':
                this.updateSystemState(data.state);
                break;
                
            case 'system_log':
                this.addSystemLog(data.message);
                break;
                
            case 'user_message':
                this.addUserMessage(data.text);
                break;
                
            case 'assistant_message':
                this.addAssistantMessage(data.text);
                break;
                
            case 'assistant_response_chunk':
                this.handleAssistantChunk(data.text);
                break;
                
            default:
                console.log('Unknown message type:', type, data);
        }
    }

    updateSystemState(state) {
        this.systemState = state;
        
        // Update status display
        const stateText = state.charAt(0).toUpperCase() + state.slice(1).replace('_', ' ');
        this.statusText.textContent = stateText;
        this.statusDot.className = `status-dot ${state}`;
        
        // Update button states
        this.updateButtonStates();
        
        // Handle recording state based on system state
        if (state === 'active' || state === 'active_listening') {
            // Stop wake word listening and start conversation recording
            this.stopWakeWordListening();
            this.startContinuousRecording();
        } else if (state === 'thinking' || state === 'speaking') {
            // During thinking and speaking, stop recording but keep wake word listening stopped
            this.stopWakeWordListening();
            this.stopContinuousRecording();
        } else if (state === 'idle') {
            // Stop conversation recording and start wake word listening
            this.stopContinuousRecording();
            if (!this.isWakeWordListening) {
                // Restart wake word listening if not already listening
                this.requestMicrophoneAndStartWakeWordListening();
            }
        } else {
            // Other states - stop all recording
            this.stopContinuousRecording();
            this.stopWakeWordListening();
        }
        
        // Update UI for current state (always call this to update indicator text)
        this.updateWakeWordListeningUI(this.isWakeWordListening);
    }

    updateButtonStates() {
        const isIdle = this.systemState === 'idle';
        const isActive = this.systemState === 'active' || this.systemState === 'active_listening';
        
        this.wakeBtn.disabled = !isIdle || !this.isConnected;
        this.sleepBtn.disabled = !isActive || !this.isConnected;
    }

    updateConnectionStatus(status) {
        this.connectionStatus.textContent = status;
        this.connectionStatus.style.color = status === 'Connected' ? '#28a745' : '#dc3545';
    }

    sendWebSocketMessage(type, data = {}) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type, data }));
        }
    }

    async wakeSystem() {
        // If we don't have microphone access yet, request it first
        if (!this.isWakeWordListening) {
            await this.requestMicrophoneAndStartWakeWordListening();
            return;
        }
        
        this.sendWebSocketMessage('wake');
        this.wakeBtn.disabled = true;
        this.wakeBtn.textContent = '🌟 Waking up...';
        
        setTimeout(() => {
            this.wakeBtn.textContent = '🌟 Wake Up ELSSA';
        }, 2000);
    }

    sleepSystem() {
        this.sendWebSocketMessage('sleep');
        this.sleepBtn.disabled = true;
        this.sleepBtn.textContent = '😴 Going to sleep...';
        
        setTimeout(() => {
            this.sleepBtn.textContent = '😴 Sleep';
        }, 2000);
    }

    async startContinuousRecording() {
        if (this.isRecording) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processAudioChunks();
            };
            
            // Start recording in chunks
            this.mediaRecorder.start(1000); // Record in 1-second chunks
            this.isRecording = true;
            
            // Show recording indicator
            this.recordingIndicator.classList.remove('hidden');
            this.audioLevel.classList.add('active');
            
            // Setup audio level visualization
            this.setupAudioVisualization(stream);
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.addSystemLog('❌ Error accessing microphone');
        }
    }

    stopContinuousRecording() {
        if (!this.isRecording) return;
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        this.isRecording = false;
        this.recordingIndicator.classList.add('hidden');
        this.audioLevel.classList.remove('active');
    }

    setupAudioVisualization(stream) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const microphone = audioContext.createMediaStreamSource(stream);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        microphone.connect(analyser);
        analyser.fftSize = 256;
        
        const updateAudioLevel = () => {
            if (!this.isRecording) return;
            
            analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            const level = Math.min(100, (average / 255) * 100);
            
            // Update audio level bars
            const bars = this.audioLevel.querySelectorAll('.level-bar');
            bars.forEach((bar, index) => {
                const barLevel = Math.max(10, (level * (index + 1)) / bars.length);
                bar.style.height = `${barLevel}%`;
            });
            
            requestAnimationFrame(updateAudioLevel);
        };
        
        updateAudioLevel();
    }

    processAudioChunks() {
        if (this.audioChunks.length === 0) return;
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
        
        // Send audio data to backend (placeholder)
        this.sendWebSocketMessage('audio_data', {
            size: audioBlob.size,
            type: audioBlob.type
        });
        
        // Clear chunks for next recording
        this.audioChunks = [];
    }

    handleAssistantChunk(text) {
        // Handle streaming response chunks
        this.currentAssistantMessage += text;
        
        // Update the last assistant message or create a new one
        const messages = this.conversationChat.querySelectorAll('.assistant-message');
        const lastMessage = messages[messages.length - 1];
        
        if (lastMessage && lastMessage.dataset.streaming === 'true') {
            // Update existing streaming message with proper line breaks
            lastMessage.querySelector('p').innerHTML = this.currentAssistantMessage.replace(/\n/g, '<br>');
        } else {
            // Create new streaming message
            this.addAssistantMessage(text, true);
        }
    }

    addChatMessage(content, type, streaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        if (streaming) {
            messageDiv.dataset.streaming = 'true';
        }
        
        const messageContent = document.createElement('p');
        // Convert \n to HTML line breaks for proper display
        messageContent.innerHTML = content.replace(/\n/g, '<br>');
        messageDiv.appendChild(messageContent);
        
        this.conversationChat.appendChild(messageDiv);
        this.scrollChatToBottom();
        
        return messageDiv;
    }

    addSystemLog(content) {
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const timestamp = document.createElement('span');
        timestamp.className = 'timestamp';
        timestamp.textContent = `[${new Date().toLocaleTimeString()}]`;
        
        const message = document.createElement('span');
        message.className = 'message';
        message.textContent = content;
        
        logEntry.appendChild(timestamp);
        logEntry.appendChild(message);
        
        this.systemLogs.appendChild(logEntry);
        this.scrollLogsToBottom();
    }

    addUserMessage(content) {
        this.addChatMessage(content, 'user-message');
    }

    addAssistantMessage(content, streaming = false) {
        if (!streaming) {
            this.currentAssistantMessage = '';
        }
        return this.addChatMessage(content, 'assistant-message', streaming);
    }

    addSystemMessage(content) {
        this.addChatMessage(content, 'system-message');
    }

    clearSystemLogs() {
        this.systemLogs.innerHTML = '';
        this.addSystemLog('🧹 System logs cleared');
    }

    clearConversationChat() {
        this.conversationChat.innerHTML = '<div class="welcome-message"><p>👋 Conversation cleared. Keep chatting!</p></div>';
    }

    scrollLogsToBottom() {
        this.systemLogs.scrollTop = this.systemLogs.scrollHeight;
    }

    scrollChatToBottom() {
        this.conversationChat.scrollTop = this.conversationChat.scrollHeight;
    }

    // Handle page visibility changes to manage recording
    handleVisibilityChange() {
        if (document.hidden && this.isRecording) {
            // Optionally pause recording when tab is not visible
            console.log('Page hidden, continuing recording...');
        } else if (!document.hidden && this.systemState === 'active' && !this.isRecording) {
            // Resume recording when tab becomes visible
            this.startContinuousRecording();
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const elssaClient = new ELSSAWebSocketClient();
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        elssaClient.handleVisibilityChange();
    });
    
    // Handle beforeunload to clean up resources
    window.addEventListener('beforeunload', () => {
        if (elssaClient.isRecording) {
            elssaClient.stopContinuousRecording();
        }
        if (elssaClient.websocket) {
            elssaClient.websocket.close();
        }
    });
});
