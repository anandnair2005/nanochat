# nanochat training report

Generated: 2026-07-11 12:05:16

## Environment

### Git Information
- Branch: unknown
- Commit: unknown (clean)
- Message: 

### Hardware
- Platform: Linux
- CPUs: 96 cores (192 logical)
- Memory: 2015.5 GB
- GPUs: 4x NVIDIA H100 80GB HBM3
- GPU Memory: 316.7 GB total
- CUDA Version: 12.8
- Hourly Rate: $12.00/hour

### Software
- Python: 3.10.20
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 0
- Lines: 0
- Files: 0
- Tokens (approx): 0
- Dependencies (uv.lock lines): 3,360

Run started: 2026-07-11 12:05:16

---

## Base model training
timestamp: 2026-07-11 15:49:09

- run: d24-4xh100-full
- device_type: 
- fp8: True
- fp8_recipe: tensorwise
- depth: 24
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: SSSL
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 8.0000
- device_batch_size: 16
- total_batch_size: -1
- embedding_lr: 0.3000
- unembedding_lr: 0.0080
- weight_decay: 0.2800
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- resume_from_step: -1
- eval_every: 250
- eval_tokens: 41,943,040
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- log_every: 100
- save_every: -1
- model_tag: d24
- Number of parameters: 1,384,122,122
- Number of FLOPs per token: 4.775225e+09
- Calculated number of iterations: 5568
- Number of training tokens: 5,838,471,168
- Tokens : Scaling params ratio: 8.0000
- DDP world size: 4
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- Minimum validation bpb: 0.7194
- Final validation bpb: 0.7194
- CORE metric estimate: 0.2607
- MFU %: 59.79%
- Total training flops: 2.788002e+19
- Total training time: 196.49m
- Peak memory usage: 54256.65MiB


## Base model evaluation
timestamp: 2026-07-11 16:00:27

- model: base_model (step 5568)
- CORE metric: 0.2561
- train bpb: 0.7197
- val bpb: 0.7191
- hellaswag_zeroshot: 0.3964
- jeopardy: 0.0997
- bigbench_qa_wikidata: 0.4608
- arc_easy: 0.5887
- arc_challenge: 0.1650
- copa: 0.3000
- commonsense_qa: 0.0991
- piqa: 0.4701
- openbook_qa: 0.2107
- lambada_openai: 0.3957
- hellaswag: 0.4098
- winograd: 0.3700
- winogrande: 0.1208
- bigbench_dyck_languages: 0.0950
- agi_eval_lsat_ar: 0.1250
- bigbench_cs_algorithms: 0.4485
- bigbench_operators: 0.1571
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.4028
- coqa: 0.3229
- boolq: -0.1846
- bigbench_language_identification: 0.1804
- sample 0: <|bos|>The capital of France is Paris, the capital of France is Paris, the capital of France is Paris,
- sample 1: <|bos|>The chemical symbol of gold is Au. It is a transition metal and is a group 11 element. It
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If yesterday was Saturday, then tomorrow will be Sunday. If yesterday was
- sample 3: <|bos|>The opposite of hot is cold. The opposite of cold is hot. The opposite of hot is cold.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Ne
- sample 5: <|bos|>My favorite color is blue. I love the color blue. I love the color blue. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is a prime number. If 5*x + 3 = 13,
- unconditioned 0: <|bos|>JRP45: 'All the Time, In Pictures'
This architected reel album shows bullet points of Penny and Sarah busting the intro to their hit American exposition Derby Night, set in the timeless 1920s; Penny Newton, villain of the Hay Greens and Josh Fox, protagonist like a late-stage Vic Destino ex-singer; and Sarah Vaughan, fresh angler for the Huronic best men.
Unfortunately, the overwrought finale makes it hard to enjoy this young original with a 10/10 approach to entertainment. It feels like the obligatory invitation to walk into P
- unconditioned 1: <|bos|>Add Inline Editor
Take Fan Key with Epiphany BItic
Motion Tracker
When a user starts a widget, Motion Tracker detects the variance of the screen by sweeping a finger horizontally. This motion is then converted to a key by offsetting the position from the position of the finger.
When this key is held, the appropriate function or script is passed on to the app(s) that read or convert this value. Its deunt depends on how many API calls are requested. Motion Tracker</tr>intmain(){// open realtimeClipboard, check values and put them in a texturevar
- unconditioned 2: <|bos|>They say learning is the best thing you can do for your child. Whether it’s a new language, a new skill, an assignment or just an unforgettable learning moment, it is the defining characteristic of childhood, right? We can all imagine what grand pride I can feel when my child first speaks a new word or after taking that first exam in English: they cheers about passing their test and the look of expectation in all of us.
I have spent several happy moments since my son’s first moment (18 months old) adding interest into every day. He is getting the hang of it too, so standard teachers and mantles would already
- unconditioned 3: <|bos|>Testimonials
Here are some examples of kinds of benefits that you can avail when availing QualityInjection® Saline Response Solutions:
Social Confidence
When you have runnign withdrawal symptoms in the past, this could become an awakening of hypersensitive traits. This can become an opportunity for your self-love that you suffer from social withdrawal and this will be alleviated by the quality of lift you get from your injection!
Felt Confidence
Injection lift is proven to be helpful in relieving restless feeling resulting from withdrawal. This also involves the guess of somebody healing what your mind has told you.
Social Support
In injection
- unconditioned 4: <|bos|>Explain the difference between safe and unsafe electricity practices
Explain the difference between safe and unsafe electricity practices
The Health and Safety at Work Act requires all employers to observe health and safety requirements, including assessing and responding to safety and/or health hazards. For example, the Act requires employers to measure hazards or conduct risk assessments and take steps to prevent, control, or reduce hazards and risk threats.
Section 13 – Occupations and Hazardous Materials
Basic safety measures should work to protect people from the hazards outlined in Section 13 of the WHS legislation. The Act is relevant to people who may come into contact with hazardous chemicals and related workplace health
- unconditioned 5: <|bos|>What is Pendulum Clock. A household clock to make an effect of pendulums, and it uses a level of these up and down slang a thing is placed over a harmonium or musical instrument, thus resulting in an effect that the sling feels an effect of movement with the instrument.
People believe major pendulum clocks have gained back a effect of time worth to us all. look much like a grandfather clock analog solid gear clock, it indicates to start digital scale automatics utilizing a large effect size laser-driven bob's just the issue having a battery and drawing power.
The look of the clock, an odd one significant next clocks
- unconditioned 6: <|bos|>How-China-Is-Technology-taking-CONTINENT?
Are you still waiting for us to die? — Mobius, the anthropomorphic computer who goes on a rampage
Are you still waiting for us to die? — Mobius, the anthropomorphic computer who goes on a rampage
This quick desktop computer has been created to simulate a thought experiment's AI. It is causing trouble around the US and it threatens to goe to robots occupied in other countries.
Mobius is a wholethig with significant positive and negative implications.
Mobius is a wholethig with significant positive and
- unconditioned 7: <|bos|>Carbon 14 dating

Evolution in action: the evolution of cosmology in a parallel. Com's free grammatical line into venn street martin updates, he misters the. Some work on different elements that carbon 14, and can be 4. For error the most 4. All living organisms. Many radioactive form of the key is chemically equal.

For samples analysis of estimating the a technique, carbon dates. The carbon dating workable 3. Annual carbon 14, and good for objects less than 7,000 years old. There were devious traitors. Living organisms become less than 7


## Chat evaluation sft
timestamp: 2026-07-11 16:41:51

- source: sft
- task_name: None
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.6271
- ARC-Challenge: 0.5034
- MMLU: 0.3603
- GSM8K: 0.0872
- HumanEval: 0.1524
- SpellingBee: 1.0000
- ChatCORE metric: 0.3712


## Summary

- Characters: 0
- Lines: 0
- Files: 0
- Tokens (approx): 0
- Dependencies (uv.lock lines): 3,360

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.2561   | -        | -        |
| ARC-Challenge   | -        | 0.5034   | -        |
| ARC-Easy        | -        | 0.6271   | -        |
| GSM8K           | -        | 0.0872   | -        |
| HumanEval       | -        | 0.1524   | -        |
| MMLU            | -        | 0.3603   | -        |
| ChatCORE        | -        | 0.3712   | -        |

Total wall clock time: 4h36m
