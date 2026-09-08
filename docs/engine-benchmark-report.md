# Engine benchmark

Generated 2026-09-05 22:32.

Every configuration ran through the app's own transcribe path with the saved settings (VAD, local pyannote speaker separation for the local and MAI engines). Where a recording has a reference transcript, the first block of rows scores against it: WER after Whisper's English normaliser, proper-noun recall, speaker attribution on aligned words, and diarization error rate. The agreement tables at the end of each section measure distance between engines, not accuracy.

## ES2004b.Array1-01

Duration: 39.1 min. Reference: AMI ES2004b.

| Attribute | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Word error rate | 0.14 | 0.17 | 0.15 | 0.17 | 0.21 | 0.23 |
| Substitutions | 255 | 229 | 377 | 334 | 282 | 283 |
| Deletions | 568 | 815 | 403 | 726 | 1093 | 1207 |
| Insertions | 110 | 98 | 238 | 121 | 57 | 75 |
| Proper-noun recall | 0.73 | 0.73 | 0.82 | 0.71 | 0.62 | 0.62 |
| Proper nouns in reference | 34 | 34 | 34 | 34 | 34 | 34 |
| Speaker attribution accuracy | 0.94 | 0.95 | 0.96 | 0.59 | 0.96 | 0.96 |
| Diarization error rate | 0.20 | 0.23 | 0.24 | 0.50 | 0.22 | 0.23 |
| Speakers in reference | 4 | 4 | 4 | 4 | 4 | 4 |
| Speaker count error | 2 | 2 | 2 | 1 | 2 | 2 |
| Words during reference silence | 64 | 58 | 87 | 44 | 35 | 35 |
| Wall time (s) | 167.20 | 72.10 | 79.40 | 98.80 | 204.30 | 230.70 |
| Audio min per wall min | 14.03 | 32.54 | 29.54 | 23.73 | 11.48 | 10.17 |
| Words | 6294 | 5856 | 6626 | 6121 | 5551 | 5463 |
| Segments | 214 | 221 | 556 | 221 | 179 | 175 |
| Speech coverage | 0.88 | 0.82 | 0.89 | 0.85 | 0.83 | 0.82 |
| Speakers | 6 | 6 | 6 | 5 | 6 | 6 |
| Speaker turns | 167 | 151 | 520 | 168 | 134 | 125 |
| Voiceprint names | - | - | - | - | - | - |
| Glossary terms found | 3 | 3 | 3 | 3 | 2 | 3 |
| Fillers (um, uh, hmm) | 171 | 2 | 209 | 154 | 6 | 7 |
| Numbers | 45 | 40 | 29 | 29 | 28 | 32 |
| Repeated adjacent segments | 0 | 0 | 6 | 1 | 0 | 0 |
| Most repeated 5-gram (count) | 2 | 3 | 2 | 2 | 3 | 2 |
| Segments with confidence | 0 | 0 | 556 | 0 | 179 | 175 |
| Mean confidence | - | - | 0.96 | - | 0.84 | 0.83 |
| Low-confidence segments | 0 | 0 | 3 | 0 | 11 | 9 |
| Retries | 0 | 0 | 0 | 0 | 0 | 0 |
| Voice embeddings returned | 6 | 6 | 0 | 0 | 6 | 6 |

Proper nouns missed:

- by every engine: Box, God, It'll
- mai-verbatim only: Designer, It, Videoplus, William's
- mai-clean only: Designer, It, Sure, William's, Yeah
- elevenlabs-scribe only: Designer, Videoplus
- gemini-verbatim only: It, Sure, Videoplus, William's, Yeah
- whisper-large-v3-turbo only: African, Designer, It, Sure, Videoplus, William's, Yeah, You
- whisper-large-v3 only: Designer, It, Sure, Videoplus, William's, Yeah, You

Word error rate of each row against each column (lower means closer):

| | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 | mean |
|---|---|---|---|---|---|---|---|
| mai-verbatim | - | 0.085 | 0.108 | 0.098 | 0.156 | 0.173 | 0.124 |
| mai-clean | 0.091 | - | 0.184 | 0.132 | 0.103 | 0.125 | 0.127 |
| elevenlabs-scribe | 0.103 | 0.163 | - | 0.145 | 0.211 | 0.227 | 0.170 |
| gemini-verbatim | 0.100 | 0.126 | 0.157 | - | 0.158 | 0.174 | 0.143 |
| whisper-large-v3-turbo | 0.177 | 0.109 | 0.252 | 0.174 | - | 0.102 | 0.163 |
| whisper-large-v3 | 0.199 | 0.134 | 0.275 | 0.195 | 0.104 | - | 0.181 |

Glossary terms not found by every engine:

| Term | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Mark | yes | yes | yes | yes |  | yes |

## EN2002a.Array1-01

Duration: 35.7 min. Reference: AMI EN2002a.

| Attribute | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Word error rate | 0.26 | 0.31 | 0.27 | 0.33 | 0.40 | 0.39 |
| Substitutions | 442 | 378 | 831 | 760 | 449 | 423 |
| Deletions | 1557 | 2032 | 1075 | 1688 | 2615 | 2622 |
| Insertions | 71 | 53 | 246 | 174 | 41 | 42 |
| Proper-noun recall | 0.72 | 0.61 | 0.75 | 0.65 | 0.52 | 0.49 |
| Proper nouns in reference | 75 | 75 | 75 | 75 | 75 | 75 |
| Speaker attribution accuracy | 0.87 | 0.88 | 0.81 | 0.70 | 0.89 | 0.88 |
| Diarization error rate | 0.40 | 0.42 | 0.50 | 0.54 | 0.45 | 0.43 |
| Speakers in reference | 4 | 4 | 4 | 4 | 4 | 4 |
| Speaker count error | 0 | 0 | 1 | 1 | 0 | 0 |
| Words during reference silence | 0 | 0 | 9 | 0 | 1 | 4 |
| Wall time (s) | 145.90 | 65.70 | 33.90 | 110.30 | 196.10 | 217.10 |
| Audio min per wall min | 14.69 | 32.60 | 63.16 | 19.42 | 10.93 | 9.87 |
| Words | 6114 | 5578 | 6799 | 6080 | 5004 | 5007 |
| Segments | 485 | 445 | 909 | 523 | 400 | 404 |
| Speech coverage | 0.82 | 0.79 | 0.80 | 0.79 | 0.77 | 0.80 |
| Speakers | 4 | 4 | 5 | 5 | 4 | 4 |
| Speaker turns | 450 | 396 | 880 | 474 | 360 | 370 |
| Voiceprint names | - | - | - | - | - | - |
| Glossary terms found | 3 | 3 | 4 | 3 | 3 | 3 |
| Fillers (um, uh, hmm) | 71 | 3 | 124 | 63 | 5 | 9 |
| Numbers | 39 | 18 | 18 | 12 | 19 | 14 |
| Repeated adjacent segments | 5 | 1 | 9 | 8 | 1 | 0 |
| Most repeated 5-gram (count) | 3 | 3 | 3 | 3 | 3 | 3 |
| Segments with confidence | 0 | 0 | 909 | 0 | 400 | 404 |
| Mean confidence | - | - | 0.95 | - | 0.76 | 0.78 |
| Low-confidence segments | 0 | 0 | 4 | 0 | 47 | 34 |
| Retries | 0 | 0 | 0 | 0 | 0 | 0 |
| Voice embeddings returned | 4 | 4 | 0 | 0 | 4 | 4 |

Proper nouns missed:

- by every engine: Buccleuch, February, For, Java, Monday, Okay, Or, We're, Yeah
- mai-verbatim only: Can, GUI, Well, You
- mai-clean only: Can, In, Sure, That, Well, Whoops, You
- elevenlabs-scribe only: GUI, If, That
- gemini-verbatim only: Can, Java's, Know, Mine, Well
- whisper-large-v3-turbo only: Can, GUI, In, Is, Know, Like, Mine, Sure, That, Well, Whoops, You
- whisper-large-v3 only: But, Can, GUI, In, Is, Like, Sure, That, Well, Whoops, You

Word error rate of each row against each column (lower means closer):

| | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 | mean |
|---|---|---|---|---|---|---|---|
| mai-verbatim | - | 0.124 | 0.224 | 0.224 | 0.262 | 0.261 | 0.219 |
| mai-clean | 0.136 | - | 0.324 | 0.260 | 0.209 | 0.202 | 0.226 |
| elevenlabs-scribe | 0.201 | 0.266 | - | 0.284 | 0.353 | 0.352 | 0.291 |
| gemini-verbatim | 0.226 | 0.238 | 0.318 | - | 0.312 | 0.310 | 0.281 |
| whisper-large-v3-turbo | 0.320 | 0.233 | 0.480 | 0.379 | - | 0.203 | 0.323 |
| whisper-large-v3 | 0.319 | 0.225 | 0.477 | 0.376 | 0.203 | - | 0.320 |

Glossary terms not found by every engine:

| Term | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Mark |  |  | yes |  |  |  |

## 4474506

Duration: 66.3 min. Reference: Earnings-22 4474506.

| Attribute | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Word error rate | 0.07 | 0.11 | 0.07 | 0.08 | 0.11 | 0.11 |
| Substitutions | 229 | 248 | 223 | 277 | 303 | 279 |
| Deletions | 79 | 628 | 55 | 124 | 613 | 606 |
| Insertions | 465 | 330 | 509 | 478 | 333 | 386 |
| Proper-noun recall | 0.92 | 0.91 | 0.93 | 0.93 | 0.80 | 0.79 |
| Proper nouns in reference | 205 | 205 | 205 | 205 | 205 | 205 |
| Speaker attribution accuracy | 0.99 | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 |
| Diarization error rate | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 |
| Speakers in reference | 17 | 17 | 17 | 17 | 17 | 17 |
| Speaker count error | -2 | -2 | -2 | 1 | -2 | -2 |
| Words during reference silence | 114 | 93 | 116 | 102 | 95 | 92 |
| Wall time (s) | 262.10 | 83.00 | 44.00 | 181.00 | 354.90 | 399.80 |
| Audio min per wall min | 15.17 | 47.90 | 90.46 | 21.96 | 11.20 | 9.95 |
| Words | 11868 | 10761 | 12189 | 11732 | 10768 | 10834 |
| Segments | 213 | 217 | 236 | 214 | 210 | 204 |
| Speech coverage | 0.94 | 0.92 | 0.94 | 0.94 | 0.93 | 0.94 |
| Speakers | 15 | 15 | 15 | 18 | 15 | 15 |
| Speaker turns | 126 | 118 | 141 | 125 | 123 | 125 |
| Voiceprint names | - | - | - | - | - | - |
| Glossary terms found | 7 | 8 | 8 | 8 | 7 | 8 |
| Fillers (um, uh, hmm) | 460 | 10 | 481 | 384 | 4 | 7 |
| Numbers | 335 | 266 | 31 | 262 | 258 | 250 |
| Repeated adjacent segments | 0 | 0 | 0 | 0 | 0 | 0 |
| Most repeated 5-gram (count) | 7 | 7 | 7 | 7 | 7 | 7 |
| Segments with confidence | 0 | 0 | 236 | 0 | 210 | 204 |
| Mean confidence | - | - | 0.99 | - | 0.96 | 0.94 |
| Low-confidence segments | 0 | 0 | 0 | 0 | 0 | 2 |
| Retries | 0 | 0 | 0 | 0 | 0 | 0 |
| Voice embeddings returned | 15 | 15 | 0 | 0 | 15 | 15 |

Proper nouns missed:

- by every engine: Berkeley, Champaign, Costco's, Heinbockel, Melich, Room, There's, Tim
- mai-verbatim only: Christmas, City, FX, Guttman, Innovel, Jeffrey, Peterman
- mai-clean only: Christmas, City, FX, Guttman, Innovel, Jeffrey, Okay, Omicron, Peterman
- elevenlabs-scribe only: Christmas, City, Guttman, Innovel, Jeffrey, US
- gemini-verbatim only: Christmas, Horvers, Jeffrey, Peterman, Steph, There
- whisper-large-v3-turbo only: Chuck, City, Costco, FX, Guttman, Horvers, Innovel, KS, Okay, Omicron, Oppenheimer, Peterman, US
- whisper-large-v3 only: Bania, Christmas, City, Costco, FX, Guttman, Horvers, Innovel, Jeffrey, KS, Okay, Omicron, There, US, Year's

Word error rate of each row against each column (lower means closer):

| | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 | mean |
|---|---|---|---|---|---|---|---|
| mai-verbatim | - | 0.112 | 0.083 | 0.057 | 0.123 | 0.126 | 0.100 |
| mai-clean | 0.124 | - | 0.182 | 0.125 | 0.059 | 0.066 | 0.111 |
| elevenlabs-scribe | 0.081 | 0.161 | - | 0.099 | 0.168 | 0.168 | 0.135 |
| gemini-verbatim | 0.058 | 0.115 | 0.103 | - | 0.121 | 0.120 | 0.103 |
| whisper-large-v3-turbo | 0.136 | 0.059 | 0.190 | 0.132 | - | 0.046 | 0.113 |
| whisper-large-v3 | 0.138 | 0.066 | 0.189 | 0.130 | 0.045 | - | 0.114 |

Glossary terms not found by every engine:

| Term | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Chris | yes | yes | yes | yes |  | yes |
| Plus Five |  | yes | yes | yes | yes | yes |

## 4483937

Duration: 60.9 min. Reference: Earnings-22 4483937.

| Attribute | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Word error rate | 0.08 | 0.10 | 0.08 | 0.09 | 0.10 | 0.10 |
| Substitutions | 209 | 198 | 214 | 228 | 248 | 239 |
| Deletions | 105 | 307 | 83 | 141 | 280 | 263 |
| Insertions | 357 | 324 | 393 | 349 | 305 | 347 |
| Proper-noun recall | 0.85 | 0.84 | 0.87 | 0.85 | 0.82 | 0.83 |
| Proper nouns in reference | 217 | 217 | 217 | 217 | 217 | 217 |
| Speaker attribution accuracy | 1.00 | 1.00 | 0.98 | 0.73 | 1.00 | 1.00 |
| Diarization error rate | 0.06 | 0.07 | 0.08 | 0.32 | 0.07 | 0.07 |
| Speakers in reference | 4 | 4 | 4 | 4 | 4 | 4 |
| Speaker count error | 0 | 0 | 2 | 2 | 0 | 0 |
| Words during reference silence | 80 | 72 | 90 | 80 | 74 | 79 |
| Wall time (s) | 258.90 | 80.10 | 33.70 | 140.70 | 284.30 | 296.30 |
| Audio min per wall min | 14.11 | 45.59 | 108.50 | 25.97 | 12.85 | 12.33 |
| Words | 9113 | 8342 | 9340 | 8627 | 8397 | 8525 |
| Segments | 139 | 161 | 138 | 148 | 139 | 122 |
| Speech coverage | 0.91 | 0.89 | 0.91 | 0.89 | 0.91 | 0.92 |
| Speakers | 4 | 4 | 6 | 6 | 4 | 4 |
| Speaker turns | 51 | 51 | 50 | 44 | 51 | 51 |
| Voiceprint names | - | - | - | - | - | - |
| Glossary terms found | 6 | 6 | 6 | 6 | 6 | 6 |
| Fillers (um, uh, hmm) | 549 | 0 | 585 | 79 | 54 | 102 |
| Numbers | 101 | 91 | 11 | 92 | 91 | 86 |
| Repeated adjacent segments | 0 | 0 | 0 | 0 | 0 | 0 |
| Most repeated 5-gram (count) | 5 | 5 | 5 | 5 | 5 | 4 |
| Segments with confidence | 0 | 0 | 138 | 0 | 139 | 122 |
| Mean confidence | - | - | 0.99 | - | 0.95 | 0.93 |
| Low-confidence segments | 0 | 0 | 0 | 0 | 1 | 1 |
| Retries | 0 | 0 | 0 | 0 | 0 | 0 |
| Voice embeddings returned | 4 | 4 | 0 | 0 | 4 | 4 |

Proper nouns missed:

- by every engine: AM, Allen, Allen's, ENT, Hawkline, Hurn, Okay, Paige, Sir, W-
- mai-verbatim only: Davis, Deer, Equator, Meggitt, Pockett, Renishaw, Renishaw's, Will, Yeah
- mai-clean only: Davis, Deer, Equator, Meggitt, Renishaw, Renishaw's, Robert, Will, Yeah
- elevenlabs-scribe only: David, Davis, Deer, Plom, Pockett, Renishaw's, Robert
- gemini-verbatim only: Equator, Meggitt, Plom, Pockett, Renishaw, Robert, Will, Yeah
- whisper-large-v3-turbo only: Deer, Equator, Jha, Jones, Meggitt, Miskin, NI, Pockett, Renishaw, Renishaw's, Robert, Will, Will-, Yeah
- whisper-large-v3 only: EV, Jha, Jones, Look, Meggitt, NI, Plom, Renishaw, Robert, Will, Will-, Yeah

Word error rate of each row against each column (lower means closer):

| | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 | mean |
|---|---|---|---|---|---|---|---|
| mai-verbatim | - | 0.092 | 0.062 | 0.085 | 0.097 | 0.097 | 0.087 |
| mai-clean | 0.101 | - | 0.156 | 0.061 | 0.034 | 0.052 | 0.081 |
| elevenlabs-scribe | 0.060 | 0.140 | - | 0.123 | 0.137 | 0.137 | 0.119 |
| gemini-verbatim | 0.090 | 0.059 | 0.133 | - | 0.066 | 0.070 | 0.084 |
| whisper-large-v3-turbo | 0.105 | 0.033 | 0.153 | 0.068 | - | 0.049 | 0.082 |
| whisper-large-v3 | 0.103 | 0.051 | 0.150 | 0.071 | 0.048 | - | 0.085 |

## ES2004b.Mix-Headset

Duration: 39.1 min. Reference: AMI ES2004b.

| Attribute | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 |
|---|---|---|---|---|---|---|
| Word error rate | 0.12 | 0.15 | 0.13 | 0.15 | 0.21 | 0.20 |
| Substitutions | 236 | 232 | 310 | 295 | 254 | 261 |
| Deletions | 462 | 710 | 320 | 577 | 1114 | 1031 |
| Insertions | 108 | 82 | 228 | 153 | 43 | 59 |
| Proper-noun recall | 0.79 | 0.77 | 0.82 | 0.71 | 0.62 | 0.62 |
| Proper nouns in reference | 34 | 34 | 34 | 34 | 34 | 34 |
| Speaker attribution accuracy | 0.95 | 0.97 | 0.97 | 0.78 | 0.97 | 0.97 |
| Diarization error rate | 0.17 | 0.20 | 0.21 | 0.33 | 0.21 | 0.20 |
| Speakers in reference | 4 | 4 | 4 | 4 | 4 | 4 |
| Speaker count error | 2 | 2 | 0 | 0 | 2 | 2 |
| Words during reference silence | 11 | 9 | 32 | 4 | 8 | 8 |
| Wall time (s) | 185.10 | 73.50 | 30.50 | 106.10 | 197.90 | 194.00 |
| Audio min per wall min | 12.67 | 31.90 | 76.80 | 22.10 | 11.85 | 12.09 |
| Words | 6416 | 5939 | 6720 | 6292 | 5533 | 5631 |
| Segments | 255 | 247 | 576 | 306 | 187 | 201 |
| Speech coverage | 0.87 | 0.82 | 0.85 | 0.86 | 0.83 | 0.84 |
| Speakers | 6 | 6 | 4 | 4 | 6 | 6 |
| Speaker turns | 211 | 182 | 530 | 244 | 148 | 156 |
| Voiceprint names | - | - | - | - | - | - |
| Glossary terms found | 3 | 3 | 3 | 3 | 3 | 3 |
| Fillers (um, uh, hmm) | 178 | 0 | 229 | 148 | 18 | 15 |
| Numbers | 42 | 33 | 27 | 30 | 33 | 29 |
| Repeated adjacent segments | 2 | 1 | 3 | 5 | 0 | 0 |
| Most repeated 5-gram (count) | 3 | 2 | 2 | 2 | 2 | 2 |
| Segments with confidence | 0 | 0 | 576 | 0 | 187 | 201 |
| Mean confidence | - | - | 0.98 | - | 0.87 | 0.86 |
| Low-confidence segments | 0 | 0 | 2 | 0 | 8 | 7 |
| Retries | 0 | 0 | 0 | 0 | 0 | 0 |
| Voice embeddings returned | 6 | 6 | 0 | 0 | 6 | 6 |

Proper nouns missed:

- by every engine: Box, God, It'll, Videoplus
- mai-verbatim only: It, William's
- mai-clean only: It, Sure, William's
- elevenlabs-scribe only: Instruction
- gemini-verbatim only: Instruction, It, Sure, William's
- whisper-large-v3-turbo only: Designer, Instruction, It, Sure, William's, You
- whisper-large-v3 only: Designer, It, Marketing, Sky, Sure, William's, Yeah, You

Word error rate of each row against each column (lower means closer):

| | mai-verbatim | mai-clean | elevenlabs-scribe | gemini-verbatim | whisper-large-v3-turbo | whisper-large-v3 | mean |
|---|---|---|---|---|---|---|---|
| mai-verbatim | - | 0.094 | 0.088 | 0.091 | 0.167 | 0.162 | 0.120 |
| mai-clean | 0.102 | - | 0.168 | 0.121 | 0.113 | 0.109 | 0.123 |
| elevenlabs-scribe | 0.084 | 0.149 | - | 0.128 | 0.212 | 0.208 | 0.156 |
| gemini-verbatim | 0.093 | 0.114 | 0.136 | - | 0.164 | 0.155 | 0.132 |
| whisper-large-v3-turbo | 0.194 | 0.121 | 0.258 | 0.186 | - | 0.104 | 0.173 |
| whisper-large-v3 | 0.185 | 0.115 | 0.248 | 0.174 | 0.102 | - | 0.165 |
