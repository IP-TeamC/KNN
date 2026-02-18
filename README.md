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

In den meisten Fällen fängt man damit an, seine Daten in eine Form zu bringen, in der
das KNN damit arbeiten kann. Im besten Fall liegen die Daten in einer CSV-Datei vor, sodass
man mithilfe des `CSV-Readers` und der Angabe des Dateipfades mit Input und Output Größen
diese in ein `Dataset` umwandeln kann, welches von den `network`-Klassen akzeptiert wird. 
In Vorbereitung für das Training lässt sich aus diesem Dataset dann durch `shuffleAndSplit`
ein Trainings- und ein Testsplit erstellen. Hier muss nur der Random-Seed und der 
Split-Anteil eingegeben werden.

Nun können sog. `DenseLayer` erstellt werden. Hier müssen 2 Aktivierungsfunktionen übergeben
und pro gewünschtem Layer ein Integer-Wert eingegeben werden. Die beiden Aktivierungsfunktionen
simulieren einmal die Funktion im Neuron selbst und einmal die Aktivierungsfunktion am Ende des
Neurons. Unter `network/activation` lassen sich alle Verfügbaren Funktionen finden. 