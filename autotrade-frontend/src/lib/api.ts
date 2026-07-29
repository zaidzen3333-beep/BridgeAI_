// ─────────────────────────────────────────────────────────────────
// API Service Layer — Aligned with FastAPI SSE Backend
//
// The backend streams Server-Sent Events for /chat and /audio-chat/stream.
// This module provides:
//   1. Helper types for the SSE event payloads
//   2. streamChat()      — SSE streaming for text chat
//   3. streamAudioChat() — SSE streaming for voice chat
//   4. getHistory()      — JSON fetch for chat history
//   5. getRecentSessions() / clearHistory() — unchanged helpers
// ─────────────────────────────────────────────────────────────────

export const API_URL = 'http://localhost:8000';

// ── SSE Event Types emitted by the backend ──
export interface SSETokenEvent {
  type: 'token';
  content: string;
}

export interface SSEStatusEvent {
  type: 'status';
  content: string;
}

export interface SSEDoneEvent {
  type: 'done';
  sources: string[];
  agents: string;
}

export interface SSETranscriptionEvent {
  type: 'transcription';
  content: string;
}

export type SSEEvent = SSETokenEvent | SSEStatusEvent | SSEDoneEvent | SSETranscriptionEvent;

// ── Callbacks for the streaming consumer ──
export interface StreamCallbacks {
  onToken: (text: string) => void;
  onStatus: (status: string) => void;
  onDone: (sources: string[], agents: string) => void;
  onTranscription?: (text: string) => void;
  onError: (error: string) => void;
}

// ── Internal SSE line parser ──
async function consumeSSEStream(response: Response, callbacks: StreamCallbacks) {
  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError('Response body is empty — no stream available.');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  // Stateful filter to strip <think>...</think> blocks across chunked tokens
  let inThinking = false;
  let thinkBuf = '';

  const emitCleanToken = (raw: string) => {
    thinkBuf += raw;
    let clean = '';

    while (thinkBuf.length > 0) {
      if (inThinking) {
        const endIdx = thinkBuf.indexOf('</think>');
        if (endIdx !== -1) {
          thinkBuf = thinkBuf.slice(endIdx + 8);
          inThinking = false;
        } else {
          // Still inside think block — discard and wait for more
          thinkBuf = '';
          break;
        }
      } else {
        const startIdx = thinkBuf.indexOf('<think>');
        if (startIdx !== -1) {
          clean += thinkBuf.slice(0, startIdx);
          thinkBuf = thinkBuf.slice(startIdx + 7);
          inThinking = true;
        } else {
          // Check for a partial <think> tag building at the end
          const lastLt = thinkBuf.lastIndexOf('<');
          if (lastLt !== -1 && '<think>'.startsWith(thinkBuf.slice(lastLt))) {
            clean += thinkBuf.slice(0, lastLt);
            thinkBuf = thinkBuf.slice(lastLt);
            break;
          } else {
            clean += thinkBuf;
            thinkBuf = '';
            break;
          }
        }
      }
    }

    if (clean) {
      callbacks.onToken(clean);
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by double newlines
      const parts = buffer.split('\n\n');
      // The last element may be an incomplete frame — keep it in the buffer
      buffer = parts.pop() || '';

      for (const part of parts) {
        for (const line of part.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const data: SSEEvent = JSON.parse(jsonStr);

            switch (data.type) {
              case 'token':
                emitCleanToken(data.content);
                break;
              case 'status':
                callbacks.onStatus(data.content);
                break;
              case 'done':
                // Flush any remaining non-think buffer
                if (thinkBuf && !inThinking) {
                  callbacks.onToken(thinkBuf);
                  thinkBuf = '';
                }
                callbacks.onDone(data.sources || [], data.agents || '');
                break;
              case 'transcription':
                callbacks.onTranscription?.(data.content);
                break;
            }
          } catch {
            // Skip malformed JSON chunks silently
          }
        }
      }
    }
  } catch (err: any) {
    callbacks.onError(err.message || 'Stream reading failed');
  } finally {
    reader.releaseLock();
  }
}

// ─────────────────────────────────────────────
// 1. Text Chat — POST /chat (SSE stream)
// ─────────────────────────────────────────────
export async function streamChat(
  question: string,
  sessionId: string,
  callbacks: StreamCallbacks
): Promise<void> {
  const res = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });

  if (!res.ok) {
    callbacks.onError(`Server responded with ${res.status}`);
    return;
  }

  await consumeSSEStream(res, callbacks);
}

// ─────────────────────────────────────────────
// 2. Audio Transcribe — POST /audio-chat/transcribe (JSON)
// ─────────────────────────────────────────────
export async function transcribeAudio(audioBlob: Blob): Promise<string> {
  const formData = new FormData();
  formData.append('file', audioBlob, 'voice_memo.webm');

  const res = await fetch(`${API_URL}/audio-chat/transcribe`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Server responded with ${res.status}`);
  }

  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.transcription;
}

// ─────────────────────────────────────────────
// 2.b Audio Chat — POST /audio-chat/stream (SSE stream)
// ─────────────────────────────────────────────
export async function streamAudioChat(
  audioBlob: Blob,
  sessionId: string,
  callbacks: StreamCallbacks
): Promise<void> {
  const formData = new FormData();
  formData.append('file', audioBlob, 'voice_memo.webm');
  formData.append('session_id', sessionId);

  const res = await fetch(`${API_URL}/audio-chat/stream`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    callbacks.onError(`Server responded with ${res.status}`);
    return;
  }

  await consumeSSEStream(res, callbacks);
}

// ─────────────────────────────────────────────
// 3. History — GET /history/{session_id} (JSON)
// ─────────────────────────────────────────────
export async function getHistory(sessionId: string) {
  const res = await fetch(`${API_URL}/history/${sessionId}`);
  if (!res.ok) throw new Error('Failed to load history');
  return res.json();
}

export async function getPredictionHistory(userId: string) {
  const res = await fetch(`${API_URL}/api/predict/history?user_id=${userId}`);
  if (!res.ok) throw new Error('Failed to load prediction history');
  return res.json();
}

// ─────────────────────────────────────────────
// 4. Recent Sessions — GET /sessions/{userId} (JSON)
// ─────────────────────────────────────────────
export async function getRecentSessions(userId: string) {
  const res = await fetch(`${API_URL}/sessions/${userId}`);
  if (!res.ok) throw new Error('Failed to load sessions');
  return res.json();
}

// ─────────────────────────────────────────────
// 5. Clear History — DELETE /history/{session_id}
// ─────────────────────────────────────────────
export async function clearHistory(sessionId: string) {
  const res = await fetch(`${API_URL}/history/${sessionId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear history');
  return res.json();
}

// ─────────────────────────────────────────────
// 5b. Delete a chat session — DELETE /session/{session_id}
// ─────────────────────────────────────────────
export async function deleteSession(sessionId: string) {
  const res = await fetch(`${API_URL}/session/${sessionId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete session');
  return res.json();
}

// ─────────────────────────────────────────────
// 5c. Delete a prediction — DELETE /api/predict/{prediction_id}
// ─────────────────────────────────────────────
export async function deletePrediction(predictionId: number) {
  const res = await fetch(`${API_URL}/api/predict/${predictionId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete prediction');
  return res.json();
}

// ─────────────────────────────────────────────
// 5d. Routing History — GET /api/route/history + DELETE /api/route/{id}
// ─────────────────────────────────────────────
export async function getRoutingHistory(userId: string) {
  const res = await fetch(`${API_URL}/api/route/history?user_id=${userId}`);
  if (!res.ok) throw new Error('Failed to load routing history');
  return res.json();
}

export async function deleteRoutingHistory(routeId: number) {
  const res = await fetch(`${API_URL}/api/route/${routeId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete route');
  return res.json();
}

// ─────────────────────────────────────────────
// 7. Prediction Chat — POST /api/predict/chat (JSON)
// ─────────────────────────────────────────────
export interface PredictionShapCause {
  stage: 'Origin' | 'Transit' | 'Destination' | 'General';
  stage_label?: string;
  days: number;
  title: string;
  detailed_cause: string;
}

export interface PredictionData {
  delay_days: number;
  shap_causes: PredictionShapCause[];
  detailed_analysis?: string | null;
  document_warning: string | null;
  language?: string;
  ui_labels?: Record<string, string>;
  env_scores?: Record<string, unknown>;
  variables: {
    direction: string;
    transport_mode: string;
    weight: number;
    origin: string;
    destination: string;
  } | null;
  chat_history?: { role: string; content: string }[] | null;
}

export interface PredictionResponse {
  status: 'waiting_for_info' | 'success';
  message: string;
  prediction_data: PredictionData | null;
  prediction_id?: number | null;
}

export async function predictChat(
  conversationHistory: { role: string; content: string }[],
  message: string,
  userId: string,
  currentPrediction?: PredictionData | null,
  predictionId?: number | null
): Promise<PredictionResponse> {
  const res = await fetch(`${API_URL}/api/predict/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversation_history: conversationHistory,
      message,
      current_prediction: currentPrediction || null,
      user_id: userId,
      prediction_id: predictionId || null,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => null);
    throw new Error(errorData?.error || 'Failed to get prediction');
  }

  return res.json();
}

// ─────────────────────────────────────────────
// 8. Companies — GET /api/companies, POST /api/companies
// ─────────────────────────────────────────────
export async function getCompanies() {
  const res = await fetch(`${API_URL}/api/companies`);
  if (!res.ok) throw new Error('Failed to fetch companies');
  return res.json();
}

export async function createCompany(companyName: string, country: string) {
  const res = await fetch(`${API_URL}/api/companies`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ company_name: companyName, country }),
  });
  if (!res.ok) throw new Error('Failed to create company');
  return res.json();
}

// ─────────────────────────────────────────────
// 8. Vision Chat — POST /chat/vision (SSE stream, multipart/form-data)
//    Expects: { file: File, question: string, session_id: string }
// ─────────────────────────────────────────────
export async function streamVisionChat(
  file: File,
  question: string,
  sessionId: string,
  callbacks: StreamCallbacks
): Promise<void> {
  const formData = new FormData();
  formData.append('files', file);
  formData.append('question', question);
  formData.append('session_id', sessionId);

  // NOTE: Do NOT set Content-Type header — the browser must set it automatically
  // so the multipart boundary is included correctly.
  const res = await fetch(`${API_URL}/chat/documents`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    callbacks.onError(`Server responded with ${res.status}`);
    return;
  }

  await consumeSSEStream(res, callbacks);
}
