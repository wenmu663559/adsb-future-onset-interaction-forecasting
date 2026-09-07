# Execution Protocol

Every new AI/Codex conversation must assume no conversational memory.

Before executing any stage, read README, project context, project state, human decisions, this protocol, the previous stage report and decision, and the current task file.

Only one stage may be executed at a time. A stage may proceed only when the previous decision is `PROCEED_TO_Rxx`.

Do not infer missing human decisions. Do not modify raw official data or sealed test splits.

Every stage must produce code, configuration, machine-readable outputs, tests, a report, a decision JSON, and updated project state. A failed or blocked stage must preserve evidence and stop.

