# Trinity LiveCodeBench Evaluation

Use this directory to reproduce the Trinity router evaluation on LiveCodeBench from a clean machine.

## 1. Environment

```bash
conda create -n fugu python=3.11
conda activate fugu
pip install -e .
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 2. Keys

Export the API credentials required by the routers (OpenAI, Claude, Gemini) before running any scripts, for example:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## 3. Host the open model

Launch the local server that backs the open-weight model:

```bash
bash server.sh
```

## 4. Decompose weights (once)

```bash
python3 decompose_model.py
```

## 5. Run the evaluation

```bash
python3 evaluate_trinity_livecodebench.py
```
