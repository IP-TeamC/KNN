# Konfigurationsmöglichkeiten

Dieses Projekt ermöglicht es, Konfigurationsdateien zu verwenden, um die Netzwerkarchitektur, das Training, die Evaluation und die Visualisierung eines neuronalen Netzes schnell, strukturiert und reproduzierbar zu steuern.
Zusätzlich besteht die Möglichkeit des Modellimports und -exports, sodass bestehende Modelle zu einem späteren Zeitpunkt analysiert oder weitertrainiert werden können.

### Speicherort
 Konfigurationsdateien sollten im Verzeichnis `./conf` abgelegt werden.

### Dateiformat
Das unterstützte Dateiformati ist [TOML](https://toml.io/en/).

TOML erlaubt Abschnisttsdefinitionen wie bspw. `[network]` sowie wiederholbare Blöcke wie `[[network.layer]]`, die hier vor allem für die Definition mehrerer Netzwerkschichten verwendet werden. 

<br>

## Beispiele von Konfigurationen

### Beispiel 1
``` toml
[data]
file = "data/banana_quality.csv"
inputStart = 0
inputSize = 7
outputStart = 7
outputSize = 1
seed = 42
testShare = 0.2

[network]
seed = 42
weightInitializer = "GLOROT_UNIFORM"

[[network.layer]]
neurons = 50
activationFunction = "SWISH"

[[network.layer]]
neurons = 50
activationFunction = "SWISH"

[[network.layer]]
neurons = 1
activationFunction = "SIGMOID"

[trainer]
lossFunction = "CROSS_ENTROPY_LOSS"
maxEpochs = 20
shuffleEpoch = true
batchSize = 1
learningRate = 0.04
exportFile = "models/example.knn"

[scorer]
lossFunction = "CROSS_ENTROPY_LOSS"
type = "ClassificationScorer"

[visualization]
heatmapInterval = 10
sankeyInterval = 10
```
### Beispiel 2
``` toml
[data]
file = "data/banana_quality.csv"
inputStart = 0
inputSize = 7
outputStart = 7
outputSize = 1
seed = 42
testShare = 0.2

[network]
importFile = "models/example.knn"

[scorer]
type = "ClassificationScorer"
```
<br>

## Konfigurationen ausführen

### Option 1: ConfigExecutor
Im Paket `de.fhdw.knn.run` befindet sich die Klasse `ConfigExecutor`, welche interaktiv verwendet werden kann, um entsprechende Konfigurationen über die Eingabeaufforderung zu laden und auszuführen:

``` cpp
Config (.toml-Dateiendung optional): example.toml
```

### Option 2: Dedizierte Runner-Klasse
Alternativ kann eine eigene Main-Klasse definiert werden, die eine bestimmte Konfigurationsdatei lädt und ausführt:

``` java
import de.fhdw.knn.config.Config;
import java.io.IOException;

public class ConfRun {
    public static void main(String[] args) throws IOException {
        Config config = Config.read("conf/example.toml");
        config.execute();
    }
}
```
<br>

## Verfügbare Konfigurationsoptionen 

### Daten
Der Abschnitt [data] definiert die Datenquelle sowie Aufteilung in Trainings- und Testdaten.
``` toml
[data]

# Pfad zu den Trainingsdaten (CSV-Datei)
file = "data/banana_quality.csv"

# Beginn der ersten Feature-Spalte
inputStart = 0

# Anzahl der Eingabefeatures
inputSize = 7

# Index der ersten Zielspalte im Datensatz
outputStart = 7

# Anzahl der Zielwerte
outputSize = 1

# Seed für Shuffle und Test-Split
seed = 42

# Anteil der Testdaten
testShare = 0.2
```
<br>

### Netzwerk
Der Abschnitt [network] definiert die globalen Einstellungen des Netzwerkes.
``` toml
[network]

# Das Saatgut für reproduzierbare Initalisierungen. Pflanze weise. 
seed = 42

# Methode zur Initialisierung der Gewichte
#
# - "GLOROT"         : Glorot/Xavier Normalverteilung
# - "GLOROT_UNIFORM" : Glorot/Xavier Gleichverteilung
# - "HE"             : He Normalverteilung
# - "HE_UNIFORM"     : He Gleichverteilung
# - "ZERO"           : Alle Gewichte werden auf 0 gesetzt
#
weightInitializer = "GLOROT_UNIFORM"

# Optionaler Pfad zum Laden eines (trainierten) Modells
# Wird ein Pfad angegeben, wird das Netzwerk aus der gespeicherten Datei geladen.
# Da die Netzstruktur bereits definiert ist, dürfen in diesm Fall KEINE weiteren [network] Einstelungen angegeben werden.
importFile = "models/example.knn"
```

Die einzelnen Schichten werden sequentiell über wiederholbare Blöcke definiert.
Die Input-Layer wird automatisch erstellt und ist nicht mit anzugeben. Die letzte Block beschreibt folglich den Output-Layer. 
``` toml
[[network.layer]]

# Anzahl an Neuronen in der jeweiligen Schicht
neurons = 50

# Aktivierungsfunktion für die jeweilige Schicht
#
# - "SIGMOID"  : S-Kurve; Werte liegen zwischen 0 und 1. Gut für Wahrscheinlichkeiten.
# - "RELU"     : Rectified Linear Unit; gibt 0 zurück, wenn Input < 0, sonst Input.
# - "LINEAR"   : Identitätsfunktion; Ausgabe = Input. Geeignet für Regressionen.
# - "TANH"     : Hyperbolische Tangens; Werte liegen zwischen -1 und 1.
# - "SWISH"    : Glatte, nichtlineare Funktion.
# - "SOFTPLUS" : Glatte Version von ReLU; log(1 + e^x).
# - "SNAKE"    : Periodische, nichtlineare Funktion; kann komplexe Muster lernen. Oder versucht es zumindest.
# - "SIN"      : Sinusfunktion; periodisch, für spezielle Aufgaben.
#
activationFunction = "SWISH"
```
<br>

### Trainer
Der Abschnit [trainer] definiert das Lernverhalten des Netzes.
``` toml
[trainer]

# Anzuwendene Verlustfunktion
#
# - "MEAN_SQUARED_ERROR"  : Klassische Verlustfunktion für Regression, berechnet den mittleren quadratischen Fehler.
# - "CROSS_ENTROPY_LOSS"  : Binary Cross Entropy Loss; Verlustfunktion für Klassifikation, misst, wie gut die Wahrscheinlichkeiten zu den Labels passen.
# - "MEAN_ABSOLUTE_ERROR" : Alternativer Regressionsverlust, berechnet den mittleren absoluten Fehler.
#
lossFunction = "CROSS_ENTROPY_LOSS"

# Maximale Anzahl an Trainingsdurchläufen über den gesamten Datensatz
maxEpochs = 20

# Festlegung, ob die Trainingsdaten vor jeder Epoche neu gemischt werden sollen
# Empfohlen für eine bessere Generalisierung.
shuffleEpoch = true

# Batchgröße
#
# Gibt an, wie viele Trainingsbeispiele gleichzeitig verarbeitet werden, bevor die Gewichte des Netzwerks angepasst werden. 
#
# batchSize = 1 : Stochastisches Training (SGD); das Netzwerk passt die Gewichte nach jedem Beispiel an.
# batchSize > 1 : Mini-Batch Training; das Netzwerk sammelt mehrere Beispiele und passt dann die Gewichte gemeinsam an.
#
batchSize = 1

# Lernrate für die Optimierung
learningRate = 0.04

# Optionaler Pfad zum Speichern des trainierten Modells
exportFile = "models/example.knn"
``` 
<br>

### Evaluation
Der Abschnitt [scorer] definiert die Bewertungsmethode des trainierten Modells.
``` toml
[scorer]

# Die zur Wertung verwendete Verlustfunktion
#
# - "MEAN_SQUARED_ERROR"  : Klassische Verlustfunktion für Regression, berechnet den mittleren quadratischen Fehler.
# - "CROSS_ENTROPY_LOSS"  : Verlustfunktion für Klassifikation, misst, wie gut die Wahrscheinlichkeiten zu den Labels passen.
# - "MEAN_ABSOLUTE_ERROR" : Alternativer Regressionsverlust, berechnet den mittleren absoluten Fehler.
#
lossFunction = "CROSS_ENTROPY_LOSS"

# Typ der Bewertung
#
# - "ClassificationScorer" : Bewertet das Modell für Klassifikationsaufgaben.
#                            Misst, wie gut die vorhergesagten Klassen mit den tatsächlichen Labels übereinstimmen. 
#                            Nutzt die angegebene Verlustfunktion, um die Leistung zu quantifizieren.
#
type = "ClassificationScorer"
```
<br>

### Darstellung
Der Abschnitt [visualization] steuert optionale Trainingsvisualisierungen.
``` toml
[visualization]

# Bestimmt, in welchem Epochenintervall eine Gewichts-Heatmap erzeugt wird
#
# Default: 0 (deaktiviert)
heatmapInterval = 10

# Bestimmt, in welchem Epochenintervall das Sankey-Diagramm aktuallisiert wird
#
# Default: 0 (deaktiviert)
sankeyInterval = 10
```



