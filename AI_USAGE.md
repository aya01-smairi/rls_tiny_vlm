# AI usage

Describe which AI tools you used (ChatGPT, Claude, Copilot, ...), for which parts of this project, and what
you personally checked, tested, or rewrote. Required for submission — see the Project Brief, section 2.
## 1. Tools Used
- **Claude / ChatGPT / Gemini :** Used for code debugging, result interpretation, and documentation formatting.
- **GitHub Copilot :** Used for inline code completion during development.
---

## 2. Specific Scope of Assistance

### A. Terminal & Shell Command Verification
- Verified terminal and CLI commands for running unit tests properly using `PYTHONPATH=. pytest`.
- Verified Git commands to manage workspace synchronization between VS Code and Google Colab (e.g., resolving local configuration file conflicts during `git pull`).

### B. Interpretation of Experimental Results
- Assisted in analyzing and interpreting the statistical breakdown of the **E1 Blind Baseline** ablation (explaining why exact match drops from **59.7%** to **1.8%** and interpreting per-attribute accuracy behavior).
- Helped articulate the performance scaling observed in the **S1 Throughput Benchmark** (~2460 img/s on GPU T4 vs. ~52 img/s on CPU at batch size 64).

### C. Debugging & Error Resolution
- Helped diagnose and resolve execution errors during debugging (e.g., fixing `ModuleNotFoundError: No module named 'src'` and handling Git file overwrite warnings).

### D. Documentation & Structuring
- Assisted in formatting and structuring documentation.
---

## 3. Human Verification & Code Ownership

In strict adherence to the challenge rules:
1. **Human Oversight:** All suggestions, terminal commands, debugging steps, and explanations provided by Claude and Gemini were manually reviewed, tested, and validated.
2. **Core Implementation:** All model components (handwritten multi-head attention, CNN encoder, adapter, Transformer decoder, tokenizer, and evaluation scripts) were implemented, thoroughly tested, and verified independently.
3. **Interview Preparedness:** I retain 100% ownership of the submitted codebase and am fully prepared to explain, walk through, and modify any line of code live during the technical interview.