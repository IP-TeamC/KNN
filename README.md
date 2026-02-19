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
Weitere Varianten der `readFile`-Methode lassen sich unter `data/CsvReader.java` einzusehen.
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

### Network

Die Netzstruktur wird in der Network-Klasse aufgebaut. Sie setzt sich aus
mehreren Dense-Layern, einem Random-Seed und einem sog. Weight-Initializer zusammen.
````java
Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 7, denseLayers);
````

#### Weight-Initializer

Bei Erstellung eines Networks müssen am Ende die Gewichte der vorher definierten Kanten initialisiert
werden. Dafür werden diese Weight-Initializer im Network angegeben. Unter `network/connection` lassen
sich die verschiedenen Initialisierungsmethoden einsehen.

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
Die verschiedenen verwendbaren Aktivierungsfunktionen lassen sich unter `network/activation` einsehen.

#### IO

Es besteht auch die Möglichkeit, bestehende Networks zu exporten und diese dann an anderen Stellen
wieder zu importen.

````java
Export.export(network);
network = Importer.importNetwork("models/bq.knn");
````


### Trainer

Der Trainer ...
````java
LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
StopFunction stopFunction = EarlyStopping.NEVER;
OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.03));

Trainer trainer = new Trainer(network, 50, true, 1, lossFunction, stopFunction, optimizationFunction);
trainer.train(train);
````

#### Learning-Rates

#### Loss-Function

#### Stop-Criteria

#### Optimization-Function


### Scorer

macht ... funktioniert so...

````java
ClassificationScorer scorer = new ClassificationScorer(network);
ClassificationScorer.Score score = scorer.score(test);

score.print();
````


### Visualization

so funktioniert visualisierung, etc...

````java
HeatmapData heatmapData = new HeatmapData(network);
HeatmapWindow window = new HeatmapWindow();
window.showSingleMatrix("Manuelle Gewichtsmatrix", heatmapData);
````