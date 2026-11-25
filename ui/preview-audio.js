/* Job-local playback from native WAV parts. The final WAV remains the source for replay and edits. */
(() => {
  'use strict';
  const RATE = 44100, LEAD = 0.025, MAX_AHEAD = 5;
  let context, active;

  function stop(session = active, status = 'stopped') {
    if (!session || session.stopped) return;
    session.stopped = true;
    session.abort.abort();
    for (const node of session.nodes) {
      node.onended = null;
      node.stop();
      node.disconnect();
    }
    session.nodes.clear();
    session.pending.length = 0;
    session.onStatus(status, session.id);
  }

  function arm() {
    stop();
    if (!window.AudioContext) return null;
    try {
      const armed = context ||= new AudioContext({ sampleRate: RATE });
      return armed.resume().then(() => armed.state === 'running' ? armed : null, () => null);
    } catch (_) { return null; }
  }

  function bufferTarget(session) {
    // Measure producer intervals (not prefilling, HTTP latency, or delayed client fetches).
    const base = session.rate === null ? 0.9 : 0.65 + 3 * Math.max(0, 1 - session.rate);
    return Math.min(4, base + 0.45 * session.stalls);
  }

  function pump(session) {
    if (session.stopped) return;
    const ctx = session.context;
    if (ctx.currentTime >= session.until && !session.buffering) {
      if (!session.done) session.stalls++;
      session.buffering = true;
    }
    if (!session.pending.length) {
      if (session.done && !session.nodes.size) stop(session, 'done');
      else if (session.buffering) session.onStatus('buffering', session.id);
      return;
    }
    if (session.buffering) {
      if (!session.done && session.pendingSeconds < bufferTarget(session)) {
        session.onStatus('buffering', session.id);
        return;
      }
      session.until = ctx.currentTime + LEAD;
      session.buffering = false;
      session.started = true;
      session.onStatus('playing', session.id);
    }
    for (const buffer of session.pending) {
      const node = ctx.createBufferSource();
      node.buffer = buffer;
      node.connect(ctx.destination);
      session.nodes.add(node);
      node.onended = () => {
        session.nodes.delete(node);
        node.disconnect();
        pump(session);
      };
      // All parts are contiguous; schedule them on one clock instead of playing separate elements.
      node.start(session.until);
      session.until += buffer.duration;
    }
    session.pending.length = 0;
    session.pendingSeconds = 0;
  }

  async function get(url, session) {
    const response = await fetch(url, { signal: session.abort.signal, cache: 'no-store' });
    if (!response.ok) throw new Error('Audio preview is unavailable');
    return response;
  }

  async function enqueue(session, row) {
    if (row.index !== session.offset || row.start_sample !== session.samples || row.sample_rate !== RATE
        || !Number.isSafeInteger(row.samples) || row.samples <= 0 || !Number.isFinite(row.produced_at)
        || typeof row.url !== 'string' || !row.url.startsWith('/files/')) {
      throw new Error('Audio preview parts are not contiguous');
    }
    while (!session.stopped && session.until - session.context.currentTime > MAX_AHEAD) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (session.stopped) return;
    const response = await get(row.url, session);
    const buffer = await session.context.decodeAudioData(await response.arrayBuffer());
    if (session.stopped) return;
    const seconds = row.samples / RATE;
    if (buffer.numberOfChannels !== 1 || Math.abs(buffer.duration - seconds) > 2 / RATE)
      throw new Error('Audio preview part has an invalid length or layout');

    const interval = row.produced_at - (session.lastPart ?? row.produced_at);
    if (interval > 0.01) {
      const observed = Math.min(4, seconds / interval);
      session.rate = session.rate === null ? observed : 0.65 * session.rate + 0.35 * observed;
    }
    session.lastPart = row.produced_at;
    session.offset++;
    session.samples += row.samples;
    session.pending.push(buffer);
    session.pendingSeconds += buffer.duration;
    pump(session);
  }

  async function receive(session, activation) {
    try {
      session.context = await activation;
      if (session.stopped) return;
      if (!session.context) throw new Error('Audio preview is unavailable');
      while (!session.stopped && !session.done) {
        const endpoint = '/api/jobs/' + encodeURIComponent(session.id) + '/preview?offset=' + session.offset;
        const update = await (await get(endpoint, session)).json();
        if (session.stopped) return;
        if (update.status === 'error' || update.status === 'cancelled') {
          stop(session);
          return;
        }
        for (const row of update.parts) await enqueue(session, row);
        session.done = update.done;
        if (session.done && !session.samples) throw new Error('No audio preview parts');
        pump(session);
      }
    } catch (_) {
      if (!session.stopped) {
        session.ownsFinal = session.started;
        stop(session, session.started ? 'stopped' : 'fallback');
      }
    }
  }

  window.StudioPreview = Object.freeze({
    arm,
    follow(id, activation, onStatus) {
      stop();
      const session = active = {
        id, onStatus, abort: new AbortController(), nodes: new Set(), pending: [],
        offset: 0, samples: 0, pendingSeconds: 0, until: 0, rate: null, stalls: 0,
        started: false, buffering: true, ownsFinal: true, stopped: false, done: false,
      };
      onStatus('buffering', id);
      void receive(session, activation);
    },
    stop() { stop(); },
    release() {
      const previous = context;
      context = null;
      stop();
      if (previous && previous.state !== 'closed') void previous.close().catch(() => {});
    },
    isActive(id) { return Boolean(active && active.id === id && !active.stopped); },
    ownsFinal(id) { return active?.id === id && active.ownsFinal; },
  });
})();
