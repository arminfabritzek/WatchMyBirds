# Codex Project Instructions - WatchMyBirds Inference App

## 1. Code Guidelines
- Verwende die Richtlinien aus AGENTS.md.


## 2. Interaction with Codex
- Codex muss:
  - Immer diese Datei und `AGENTS.md` beachten.
  - Nur produktive Skripte ändern, nie Dateien in `legacy/`.
  - Nach **jeder** Codeänderung automatisch die Dokumentation gemäß Abschnitt 8 aktualisieren.
- Bei Unsicherheit TODO-Kommentare hinzufügen.

---

## 7. Dependencies
- Python 3.11+
- Abhängigkeiten in `requirements.txt`.
- Bei neuen Paketen Datei aktualisieren und dokumentieren.

---

## 8. Documentation Maintenance Policy
- **Dokumentation muss nach jeder Codeänderung aktualisiert werden.**

---


### Pflicht-Updates
1. **README.md** und **PROJECT_STATUS.md**
   - Ordnerstruktur, Config-Keys und Anweisungen synchronisieren.  
   - Test- und Nutzungshinweise ergänzen, falls geändert.  

### Regeln
- Dokumentationsänderungen müssen **im selben Commit** wie Codeänderungen erfolgen.
- Keine veralteten Informationen in diesen Dateien belassen (aktualisieren).
- Bei Unklarheit → Datei komplett mit aktuellem Stand neu generieren.

---

### Additional Notes
- `README.md` und `PROJECT_STATUS.md` muss jederzeit den tatsächlichen Stand der Pipeline widerspiegeln.
- Dokumentationspflege ist Teil jedes Entwicklungs-Tasks.