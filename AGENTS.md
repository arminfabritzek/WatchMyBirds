# AGENTS.md – Guidelines for Codex and Agents

> **Zweck:** Sicherstellen, dass alle Agenten (einschließlich Codex) Codequalität und Dokumentation konsistent pflegen.

---

1. Installationshinweise
- Fehlende Python-Pakete installieren:  
  ```bash
  pip install -r requirements.txt

	•	Python-Version: 3.11+
	•	Bei neuen Abhängigkeiten: requirements.txt aktualisieren und dokumentieren.

⸻

2. Code-Style-Regeln

Sprache	Formatter / Linter	Besondere Regeln
Python	Black (88 Zeichen), UTF-8	Keine from … import *
JavaScript (ES6)	ESLint (airbnb-base)	Dateien nur in static/ verwenden

2.1 Namenskonventionen
	•	snake_case für Variablen und Funktionen
	•	PascalCase für Klassen
	•	Boolesche Variablen: is_…, has_…, should_…
	•	Schleifen-Indizes i/j nur in sehr kurzen Schleifen (< 5 Zeilen)
	•	Keine kryptischen Abkürzungen (df → pandas_dataframe)

2.2 Docstrings
	•	Triple-quoted """ direkt unter der Definition.
	•	Erste Zeile: Verb in 3. Person Singular (Deutsch).
	•	Mehrzeilig bei komplexen Funktionen:

Args:
    name (type): Beschreibung.
Returns:
    type: Beschreibung.
Raises:
    ErrorType: Wann der Fehler auftritt.

	•	Beispiel:

def lade_daten(pfad: str) -> pd.DataFrame:
    """Lädt Daten aus einer CSV-Datei."""

⸻

3. Commit-Nachrichten

	•	Format: <type>: <kurze Zusammenfassung>
	•	Typen: feat, fix, refactor, docs, style, test, build, ci
	•	Beispiel:

feat: add hyperparameter tuning with Optuna

⸻

4. Sicherheitsrichtlinien
	•	Änderungen nur an produktiven Skripten; keine Modifikation von legacy/.
	•	Tests (pytest) müssen bestehen, bevor Änderungen gemergt werden.

⸻

5. Dokumentation
	•	Nach jeder Codeänderung muss im selben Commit auch die Dokumentation aktualisiert werden:
	•	PROJECT_STATUS
	•	ggf. neue Konfigurationsparameter ergänzen


⸻

6. Konsistenz mit CODEX.md
	•	Alle Agenten müssen vor Ausführung:
	•	CODEX.md lesen (für Projektregeln)
	•	AGENTS.md lesen (für Arbeitsrichtlinien)
	•	Anweisungen aus beiden Dateien haben höchste Priorität.

⸻

7. CI/CD und Automatisierung
	•	GitHub Actions oder andere Automatisierungen dürfen nur laufen, wenn:
	•	Doku-Dateien aktuell sind.
	•	Tests erfolgreich sind.
	•	Bei fehlschlagenden Tests → Änderungen nicht automatisch mergen.

⸻

🔑 Wichtig:
	•	Dokumentationspflege ist ein fester Bestandteil jeder Entwicklungsaufgabe.
	•	Codex darf Änderungen nicht ohne diese Updates abschließen.
	•	Lies vor jedem Commit **PROJECT_STATUS.md** und **CODEX.md**, um die aktuelle Phase und Ziele zu kennen.
	•	Änderungen an der Codebasis müssen diese Ziele umsetzen.

---

