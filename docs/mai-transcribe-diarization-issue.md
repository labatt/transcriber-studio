Calling the fast transcription endpoint with `enhancedMode.model = "MAI-Transcribe-2"` and `diarization.enabled = true` fails for recordings longer than roughly 15 minutes. The same recordings transcribe successfully with diarization disabled — including a 73-minute, 70 MB file in a single request. The failure is therefore in the diarization component, not in upload size, duration, or transcription.

The errors returned do not describe this. They are a relayed `RequestTimeout` / Timeout` (HTTP 408) that reads like a network problem, an HTTP 500, and an HTTP 503 wrapping "Diarization service returned error code 400". None of these codes appear in the documented `DetailedErrorCode` enum for this API version, and no documented limit predicts the failure: every published limit (5 h / 500 MB on the quotas page,
2 h / 250 MB in the REST reference, 300 MB on the MAI page) places these files in range.

## Environment

- Endpoint: `https://eastus.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15`
- Region: `eastus` (listed as supporting "Transcribe with mai-transcribe model")
- Resource: Foundry resource for Speech in `eastus` (resource name and subscription id available privately on request)
- Auth: `Ocp-Apim-Subscription-Key`
- Client: Python `requests` 2.x with a streaming multipart body; connect timeout sized to the file, read timeout 1800 s
- Audio: MP3, mono, ~128 kbit/s, from a Plaud recorder; also a 32 kbit/s 16 kHz re-encode of the same file

## Request

```
POST .../speechtotext/transcriptions:transcribe?api-version=2025-10-15
Content-Type: multipart/form-data
  audio       = <file>
  definition  = {
    "enhancedMode": {
      "enabled": true,
      "model": "MAI-Transcribe-2",
      "modelOptions": { "timestamps": "word", "transcribeStyle": "clean" }
    },
    "diarization": { "enabled": true }
  }
```

The failing and succeeding requests differ only in the presence of the `diarization` object. No `phraseList` or `locales` were sent in the runs below.

## Results

All runs on 2026-09-04/05 (UTC), same resource, same region. "Upload" is the time for the request body to finish sending; the error arrived immediately after in every case.

| #   | audio                      | duration | size    | diarization | upload | result                                                |
| --- | -------------------------- | -------- | ------- | ----------- | ------ | ----------------------------------------------------- |
| 1   | clip                       | 25 s     | 0.4 MB  | on          | 1 s    | **200 OK** — 2 speakers                               |
| 2   | first 15 min of recording  | 15 min   | 14.4 MB | on          | 130 s  | **408** ×3 (see body A)                               |
| 3   | first 30 min of recording  | 30 min   | 28.8 MB | on          | 86 s   | **503** ×3 (see body B)                               |
| 4   | full recording, re-encoded | 73 min   | 17.5 MB | on          | 132 s  | **408**, then **500** (body C), then **503** (body B) |
| 5   | first 30 min of recording  | 30 min   | 28.8 MB | **off**     | 33 s   | **200 OK** — 85 segments                              |
| 6   | full recording, original   | 73 min   | 70.2 MB | **off**     | 141 s  | **200 OK** — 193 segments                             |

Run 4 (17.5 MB) failing while run 6 (70.2 MB) succeeds rules out file size. Runs 5 and 6 succeeding rules out duration and transcription. Only the `diarization` flag separates the failing runs from the succeeding ones.

## Error bodies

**A — HTTP 408** (runs 2 and 4, first attempt):

```
MAI service returned an error: RequestTimeout - { "error": { "code": "Timeout", "message": "The operation was timeout." } }
```

**B — HTTP 503** (run 3 all attempts; run 4 third attempt):

```
MAI service returned an error: ServiceUnavailable - {"error":{"code":"diarization_unavailable","message":"Speaker diarization service is unavailable: Diarization service returned error code 400"}}
```

**C — HTTP 500** (run 4, second attempt) — includes what looks like a trace id:

```
MAI service returned an error: InternalServerError - {"detail":"'d582ccb307fe4e5c93857111650d7224'"}
```

The bodies are prose wrapping a JSON object, not the documented JSON error shape, and the wrapper "MAI service returned an error" suggests these are relayed from the MAI backend rather than raised by the Speech front end.

## What the documentation says, and where it disagrees

- Quotas page — LLM speech and fast transcription: "Maximum audio length < 5 hours per file", "< 500 MB".
- REST reference, `audio` parameter, api-version 2025-10-15: "shorter than 2 hours in audio duration and smaller than 250 MB".
- MAI-Transcribe page: "less than 300 MB", no duration stated.
- LLM Speech feature table: lists the MAI-transcribe column with Diarization ❌, while the MAI-Transcribe page documents `diarization.enabled` as supported.
- `DetailedErrorCode` enum: contains `AudioLengthLimitExceeded` but no timeout-shaped code.

A 15-minute, 14 MB file is inside every one of these limits.

## Asks

1. Is there a duration limit specific to diarization in enhanced mode? If so, please document it on the MAI-Transcribe page and return `AudioLengthLimitExceeded` (or a diarization-specific code) rather than a relayed 408/500/503.
2. Please reconcile the four conflicting limit statements above.
3. Please confirm whether the ❌ under Diarization in the LLM Speech feature table for MAI-transcribe is stale or accurate.

## Workaround in use

Transcribe with `diarization` disabled — which works at full length — and run speaker diarization locally on the returned word timings.
