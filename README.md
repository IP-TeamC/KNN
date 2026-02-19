# Projekt2

### Parameter
- Anzahl Neuronen +
- Anzahl Schichten +
- Verknüpfung (vollständig?) +
- Topologie-Datei am Anfang laden +
- Aktivierungsfunktionen (spezifisch für Neuron: Eingang, Ausgang, Hidden, innerhalb Schicht?) +
- Anzahl Epochen / Abbruchkriterien +
- Early Stopping ~
- Loss Function / Verlustfunktionen ~
- Learning Rate ~
- Art der Lernfunktion: z.B. Gradient Descent (Optimierungsfunktion)
- Simulated Annealing, gefaltete Netze


# Verwendung des folgenden Projekts

## Was ist dieses Projekt?

Dieses Projekt ist eine selbst geschriebene Machine-Learning Bibliothek für zur
Erstellung und Ausführung von KNNs (künstlichen Neuronalen Netzen). Es umfasst 
das Erstellen von neuronalen Netzen mit und ohne Hidden-Layer, verschiedene 
Aktivierungs-, Verlust- und Optimierungsfunktionen, einstellbare Trainingsparameter,
Visualisierungsmethoden der KNNs und die Möglichkeit, Dateien einzulesen.

## Verwendung der Klassen

### Datenaufbereitung

#### CsvReader

Mit dem CSV-Reader ist es möglich, CSV-Dateien in ein DataSet umzuwandeln. Hier muss einmal der 
Dateipfad und jeweils der Startindex und die größe des Inputs und Outputs angegeben werden.
In der nun vorliegenden Form sind die Daten dann bereit, in Train und Test-Splits aufgeteilt zu werden.
Dabei kann ein Random-Seed und den Anteil der Daten im Test-Set angegeben werden.
````java
DataSet data = CsvReader.readFile("data/banana_quality.csv", 0, 7, 7, 1);
TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
DataSet train = trainTest.train;
DataSet test = trainTest.test;
````
#### DataSet

Als Rückgabe erhält man ein `DataSet`, welches man sich wie ein klassisches Tabellenobjekt für 
maschinelles Lernen vorstellen kann. Es enthält alle Eingabewerte und zugehörigen 
Ausgabewerte in Form von zweidimensionalen Arrays, wobei jede Zeile einem einzelnen 
Datenpunkt entspricht. Zusätzlich speichert es Metainformationen wie die Anzahl der Datensätze 
sowie die Anzahl der Ein- und Ausgabewerte pro Zeile. Darüber hinaus stellt die Klasse Methoden 
zur Vorverarbeitung bereit, etwa zum Mischen der Daten, Aufteilen in Trainings- und Testdaten 
oder zur Normalisierung.

#### Normalizer

### Network

Die Netzstruktur wird in der Network-Klasse aufgebaut. Sie setzt sich aus
mehreren Dense-Layern, einem Random-Seed und einem sog. Weight-Initializer zusammen.
````java
Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 7, denseLayers);
````

#### Weight-Initializer

Bei Erstellung eines Networks müssen am Ende die Gewichte der vorher definierten Kanten initialisiert
werden. Dafür werden diese Weight-Initializer im Network angegeben. Zu den zur Verfügung stehenden
Initializern gehören:
- `Zero-`
- `HE-`
- `HE-Uniform-`
- `Glorot-`
- `Glorot-Uniform-Weightinitializer`.

#### Input-Layer

Der Input-Layer besteht aus mehreren Input-Neuronen. Er ist so wie ein klassischer Input-Layer 
zu verstehen.

#### Dense-Layer

Der Dense-Layer bildet sowohl die Hidden-Layer als auch den Output-Layer eines neuronalen Netzes
ab. Bei der Erstellung werden eine Aktivierungsfunktion für die interne Verarbeitung innerhalb
der Neuronen sowie eine Aktivierungsfunktion für die Ausgabe definiert. Zusätzlich wird eine 
beliebige Anzahl von Integer-Werten übergeben, die jeweils die Anzahl der Neuronen pro Schicht
festlegen. Jeder dieser Integer-Werte steht für eine eigene Schicht im Netzwerk, wobei der 
letzte Wert die Größe des Output-Layers bestimmt.

````java
DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SWISH, ActivationFunction.SIGMOID, 100, 100, 1);
````
Zu den unterstützten Aktivierungsfunktionen gehören:

- `LinearActivationFunction`
- `ReLUActivationFunction`
- `SigmoidActivationFunction`
- `SinActivationFunction`
- `SnakeActivationFunction`
- `SoftplusActivationFunction`
- `SwishActivationFunction`
- `TanhActivationFunction`.

#### IO

Es besteht auch die Möglichkeit, bestehende Networks zu exporten und diese dann an anderen Stellen
wieder zu importen.

````java
Export.export(network);
network = Importer.importNetwork("models/bq.knn");
````


### Trainer

Der Trainer ist dann im endeffekt die Klasse, welche das training ausführt. Für das Training 
muss man eine vielzahl von Parametern angeben: `Network` auf dem trainiert wird, die maximale Anzahl
an `Epchen` als Integer, ein Boolean ob geshufflet werden soll, die `Batchsize`, die `Loss-Funktion`, 
die `Stopmethode` und die `Optimierungsfunktion`.
````java
LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
StopFunction stopFunction = EarlyStopping.NEVER;
OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.03));

Trainer trainer = new Trainer(network, 50, true, 1, lossFunction, stopFunction, optimizationFunction);
trainer.train(train);
````

#### Learning-Rates

Für die Steuerung der Learning-Rate stehen mehrere Strategien zur Verfügung, 
die das Trainingsverhalten des Modells beeinflussen. Zur Auswahl gehören `Constant`, `Decay`, 
`Softstart` sowie `SoftstartDecay`.

Bei `Constant` bleibt die Lernrate während des gesamten Trainingsprozesses unverändert. 
`Decay` reduziert die Lernrate schrittweise über die Trainingszeit hinweg, um gegen Ende 
stabilere und feinere Anpassungen der Gewichte zu ermöglichen. `Softstart` beginnt mit einer 
zunächst kleinen Lernrate, die in den ersten Trainingsschritten kontrolliert ansteigt, um 
ein zu starkes initiales Überschwingen zu vermeiden. `SoftstartDecay` kombiniert beide Ansätze, 
indem die Lernrate zunächst ansteigt und anschließend im weiteren Verlauf wieder 
kontinuierlich abgesenkt wird.

#### Loss-Functions

Zu jedem supervised Learning gehört auch eine Verlust-Funktion. 
Diese kann wie folgt definiert werden:

````java
LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
````
Diese Bibliothek unterstützt klassische Verlust-Funktionen:

- `Binary-Cross-Entropy-Loss`
- `Mean-Squared-Error`
- `Mean-Absolute-Error`.

#### Stop-Criteria

Diese Bibliothek unterstützt außerdem das `Early-Stopping`, um Overfitting vorzubeugen. 
Diese Stop-Funktion muss beim Erstellen des Trainers mitgegeben werden. Bislang kann man
nur Early-Stopping aktivieren und deaktivieren.

````java
StopFunction stopFunction = EarlyStopping.NEVER;
````

#### Optimization-Function

Unter Optimierungs-Funktionen ist hier die forward-Funktion zu verstehen. Also mit welchem Vorgehen
die kontinuierliche Anpassung der Gewichte passiert. Diese benötigt zur Initialisierung die ausgewählte
Verlust-Funktion und die gewählte Learning-Rate. Hier können der `Gradient Descent` und `Adjustments` 
verwendet werden.

````java
OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.03));
````


### Scorer

Der Classification-Scorer errechnet anhand eines DataSets, also in dem Fall eines Test- oder Train-Splits, 
übliche Metriken. Zu diesen Zählen die `Confusion Matrix`, die `Accuracy`, der `Error`, die `Precision`,
der `Recall` und der `F1-Score` und werden in der `Score`-Klasse zusammengefasst. Diese werden dann 
durch das print am Ende des Testings ausgegeben.

````java
ClassificationScorer scorer = new ClassificationScorer(network);
ClassificationScorer.Score score = scorer.score(test);

score.print();
````


### Visualization

Mithilfe der Networks lassen sich verschiedene Visualisierungen der Ergebnisse darstellen. Dieses
Beispiel zeigt die Erstellung einer Heatmap Windows:
````java
HeatmapData heatmapData = new HeatmapData(network);
HeatmapWindow window = new HeatmapWindow();
window.showSingleMatrix("Manuelle Gewichtsmatrix", heatmapData);
````
Networks lassen sich als `Heatmap` darstellen, in der die x- und y-Achse ein Neuron darstellt, und 
die Heatmapeinträge dann jeweils die Verbindung vom x-Neuron zum y-Neuron ist. Der `Sankey-Plot` 
stellt die Verbindungen als Linien dar. Je dicker die Linie, desto gewichteter die Verbindung.