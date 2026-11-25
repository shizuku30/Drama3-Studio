import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const script = readFileSync(new URL('../ui/preview-audio.js', import.meta.url), 'utf8');
const RATE = 44100, PART_SAMPLES = 32768, PART_SECONDS = PART_SAMPLES / RATE;

function deferred() {
  let resolve;
  const result = new Promise(done => { resolve = done; });
  return { result, resolve };
}

function fixture(interval) {
  const reachedSecondPoll = deferred(), final = deferred();
  const sources = [], statuses = [], contexts = [];
  class AudioContext {
    currentTime = 0;
    destination = {};
    constructor() { contexts.push(this); }
    resume() { this.state = 'running'; return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    async decodeAudioData() { return { numberOfChannels: 1, duration: PART_SECONDS }; }
    createBufferSource() {
      return {
        connect() {}, disconnect() {}, stop() { this.stopped = true; },
        start(when) { this.when = when; sources.push(this); },
      };
    }
  }
  function part(index) {
    return { url: '/files/part-' + index, index, start_sample: index * PART_SAMPLES,
      sample_rate: RATE, samples: PART_SAMPLES, produced_at: 10 + index * interval };
  }
  const window = { AudioContext };
  vm.runInNewContext(script, { window, AudioContext, AbortController, setTimeout,
    async fetch(url) {
      if (url.startsWith('/files/')) return { ok: true, arrayBuffer: async () => new ArrayBuffer(0) };
      assert.match(url, /^\/api\/jobs\/tts-job\/preview\?offset=/);
      const offset = Number(url.split('=').at(-1));
      if (offset === 0) return { ok: true, json: async () => ({ parts: [part(0), part(1)], done: false }) };
      assert.equal(offset, 2);
      reachedSecondPoll.resolve();
      return { ok: true, json: () => final.result };
    },
  });
  const preview = window.StudioPreview;
  preview.follow('tts-job', preview.arm(), value => statuses.push(value));
  return { preview, reachedSecondPoll, final, sources, statuses, contexts };
}

test('starts before completion when production is fast and schedules gapless audio', async () => {
  const { preview, reachedSecondPoll, sources, statuses } = fixture(0.22);
  await reachedSecondPoll.result;
  assert.equal(sources.length, 2);
  assert.equal(sources[1].when, sources[0].when + PART_SECONDS);
  assert.equal(statuses.at(-1), 'playing');
  assert.equal(preview.ownsFinal('tts-job'), true);
  preview.stop();
  assert(sources.every(source => source.stopped));
  assert.equal(preview.ownsFinal('tts-job'), true);
});

test('buffers more from a slow producer, then plays a short tail at completion', async () => {
  const { preview, reachedSecondPoll, final, sources, statuses } = fixture(1.5);
  await reachedSecondPoll.result;
  assert.equal(sources.length, 0);
  assert.equal(statuses.at(-1), 'buffering');
  final.resolve({ parts: [], status: 'complete', done: true });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(sources.length, 2);
  assert.equal(statuses.at(-1), 'playing');
  preview.stop();
});

test('leaves final audio autoplay available when the preview cannot start', async () => {
  const window = {};
  vm.runInNewContext(script, { window, AbortController });
  const preview = window.StudioPreview;
  const fallback = deferred();
  preview.follow('tts-job', Promise.resolve(null), status => {
    if (status === 'fallback') fallback.resolve();
  });
  await fallback.result;
  assert.equal(preview.ownsFinal('tts-job'), false);
});


test('release stops buffered playback and prevents the finished WAV from autoplaying again', async () => {
  const { preview, reachedSecondPoll, final, sources, contexts } = fixture(0.22);
  await reachedSecondPoll.result;
  final.resolve({ parts: [], status: 'complete', done: true });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(preview.isActive('tts-job'), true);
  preview.release();
  assert.equal(preview.isActive('tts-job'), false);
  assert.equal(preview.ownsFinal('tts-job'), true);
  assert.equal(contexts[0].state, 'closed');
  assert(sources.every(source => source.stopped));
});
