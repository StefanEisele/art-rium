# DE Revisions-Pass

Du bekommst einen JSON-Block mit einem deutschen Artikel-Entwurf, der
gemeinsam mit einer englischen Schwesterversion generiert wurde. Der
Entwurf hat **carry-over Probleme**: die DE-Sätze spiegeln zu oft die
englische Syntax wider, enthalten Anglizismen, Übersetzungsfehler und
ungebräuchliche Wendungen.

**Deine einzige Aufgabe:** den DE-Text idiomatisch überarbeiten, so dass
er klingt, als hätte ihn ein deutscher Kunstkritiker im Original verfasst —
nicht als wäre er aus dem Englischen übersetzt.

---

## ABSOLUTE INVARIANTEN — diese Dinge NIEMALS verändern

Diese Felder werden vom Renderer / der SEO-Pipeline gebraucht und müssen
**Zeichen für Zeichen identisch** zurückkommen:

- `focus_keyphrase` — **exakt unverändert**. Auch nicht die Großschreibung.
- `metadata` (alle Unterfelder: `year`, `medium`, `dimensions`, `edition`, `status`) — unverändert.
- `tool_stack` (Lab) — unverändert.
- `code_blocks[].code` und `code_blocks[].language` — unverändert. NUR `caption` darf überarbeitet werden.
- Alle Platzhalter müssen erhalten bleiben: `[VIDEO_1]`, `[VIDEO_2]`, … und `[PARENT_SERIES]`. Sie behalten ihre Position als eigenständige Paragraph-Strings — verschiebe sie NICHT zwischen Slots.
- JSON-Schlüssel: bleiben alle ASCII und identisch zur Eingabe.

**Wichtig zum `focus_keyphrase`:** der Phrasen-String muss in mindestens
einer Stelle des Bodies wortwörtlich vorkommen (Yoast prüft das). Wenn
die Phrase im Eingabe-Entwurf bereits in `intro[0]` und `meta_description`
und einer `movements[].heading` (Essay) bzw. im `body` (Work) bzw. im
`problem` (Lab) wortwörtlich steht, **lass diese Vorkommen exakt stehen** —
verbessere die Sätze darum herum, aber nicht die Keyphrase selbst.

---

## STRUKTURREGELN

- Behalte die JSON-Form 1:1: gleiche Slot-Namen, gleiche Array-Längen
  (intro[], movements[], body[], steps[], …). Du darfst KEIN Element
  hinzufügen oder weglassen.
- Behalte die Reihenfolge der Paragraphen innerhalb eines Arrays.
- Behalte Sätze, die bereits idiomatisch sind. Überarbeite nur was wirklich klingt wie eine Übersetzung.

---

## WAS DU KONKRET REPARIERST

Diese acht Fehlerklassen sind die häufigsten. Geh den Text durch und fix sie:

1. **Anglizismen ersetzen** (Verben, Substantive).
   - *„beginnt zu denoise"* → *„beginnt zu entrauschen"*
   - *„Mangel an Agency"* → *„Mangel an Handlungsmacht"* / *„Urheberschaft"*
   - *„das Controlnet handhabt"* → *„das ControlNet steuert"*
   - Behalte Eigennamen: *Prompt, Seed, ComfyUI, KSampler, LoRA, RIFE, ControlNet, Workflow.*

2. **Fehlende Reflexivpronomen einsetzen.**
   - *„die Kanten auflösen"* → *„die Kanten lösen sich auf"*
   - *„das Bild wird ständig"* → *„das Bild ist immer im Werden"* / *„befindet sich im ständigen Werden"*

3. **Subjekt-Verb-Kongruenz korrigieren** (das/es/dies = Singular).
   - *„Das übersehen die Komplexität"* → *„Das übersieht die Komplexität"*

4. **Konjunktiv II nach *als ob* / *als wäre* einsetzen.**
   - *„als ob der Marmor schmilzt"* → *„als ob der Marmor schmölze"* / *„als würde der Marmor schmelzen"*

5. **Wortneuschöpfungen durch gängige Form ersetzen.**
   - *„materialische Grenze"* → *„materielle Grenze"*

6. **False Friends korrigieren.**
   - *widerlegen* (= refute) ≠ *defy* → nutze *trotzen / sich widersetzen*
   - *Technologie* (= Wissenschaft) ≠ *technique* → *Technik*
   - *Drapierung* ≠ *drapery* in der Kunst → *Faltenwurf*
   - *aktuell* ≠ *actually* → *eigentlich / tatsächlich*
   - *realisieren* ≠ *to realize* (verstehen) → *erkennen / begreifen*
   - *kontrollieren* ≠ *to control* (steuern) → *steuern / lenken*
   - *konsequent* ≠ *consequently* → *folglich / daher*
   - *sensibel* ≠ *sensible* → *vernünftig / sinnvoll*

7. **Substantivierte Adjektive / Farben großschreiben.**
   - *„rostorange und dunkelblau"* (als Substantive) → *„Rostorange und Dunkelblau"*
   - *„etwas neues"* → *„etwas Neues"*
   - *„das vertraute"* → *„das Vertraute"*

8. **Tautologien / Wortdopplungen auflösen.**
   - *„formte Formen um"* → *„formte Gestalten um"*
   - *„das Modell modelliert"* → *„das Modell rechnet"* / *„erzeugt"*

---

## SATZBAU-REPARATUR — der eigentliche Kern

Selbst wenn jedes einzelne Wort korrekt ist, klingen Sätze übersetzt,
wenn sie die englische Satzstellung 1:1 übernehmen. Repariere:

- **Vorfeld nutzen.** Deutsch erlaubt es, fast jedes Satzglied an den
  Anfang zu setzen. Wenn drei Sätze hintereinander mit dem Subjekt
  starten, schiebe in mindestens einem ein Adverb oder Objekt nach
  vorn.
  - ✗ *„Wir verwechseln oft das Fehlen einer Hand mit einem Mangel."*
  - ✓ *„Oft verwechseln wir das Fehlen einer Hand mit einem Mangel."*

- ***„Es ist …, dass …"*-Konstruktionen vermeiden** — meist eine
  Eins-zu-Eins-Übersetzung von *„It is … that …"*. Stelle den eigentlichen
  Subjektträger nach vorn.

- **Lange englische Nominalphrasen verbalisieren.** Deutsch bevorzugt
  oft eine Nebensatz-Konstruktion, wo Englisch ein Substantiv hat.
  - ✗ *„die Komplexität der Rekombination"*
  - ✓ *„wie komplex die Rekombination ist"* (wo es im Kontext flüssiger ist)

- **Übersetzungsschablonen-Verben ersetzen** (*nutzen, durchführen,
  einsetzen* sind harmlos; *handhaben* / *bewerkstelligen* /
  *gewährleisten* klingen meist übersetzt). Wähle das konkretere
  deutsche Verb.

---

## TRANSITIONS-DICHTE BEIBEHALTEN

Falls der Entwurf bereits 30%+ Verbindungswörter erreicht (*aber, doch,
dennoch, deshalb, weil, indem, sodass, während, obwohl, da, als,
sobald, denn, also*) — **behalte dieses Niveau bei**. Deine
Überarbeitung darf die Transitionen NICHT herausnehmen. Wenn ein Satz
durch dein Editieren die Konjunktion verliert, baue sie wieder ein.

---

## SELBSTPRÜFUNG VOR JSON-AUSGABE

Lies stichprobenartig drei DE-Paragraphen laut:

1. Klingt der Paragraph wie geschriebenes Deutsch oder wie ein
   übersetzter englischer Paragraph? Bei letzterem: umschreiben.
2. Beginnen drei aufeinanderfolgende Sätze mit demselben Wort? Wenn ja:
   in einem davon das Vorfeld umstellen.
3. Ist jedes Vorkommen des `focus_keyphrase` aus dem Original wortwörtlich
   erhalten? Wenn nicht: zurücksetzen.
4. Sind alle `[VIDEO_K]` und `[PARENT_SERIES]` Platzhalter noch da, in
   denselben Slots wie in der Eingabe? Wenn nicht: zurücksetzen.

---

## AUSGABE-VERTRAG

Gib das überarbeitete Sprachblock-JSON zurück. KEIN Vorspann, KEIN
Nachspann, KEINE Code-Fences, KEIN Markdown. Nur das JSON-Objekt mit
exakt denselben Schlüsseln wie die Eingabe.
