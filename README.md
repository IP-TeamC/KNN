# KNN-Bibliothek

Eine selbst entwickelte Java-Bibliothek zum Erstellen und Trainieren von künstlichen neuronalen Netzen (KNNs), entwickelt im Zuge eines Integrationsprojekts an der Fachhochschule für die Wirtschaft Hannover.

📚 **[Zur vollständigen Dokumentation](https://ip-teamc.github.io/KNN/)**


## Was ist dieses Projekt?

Dieses Projekt ist eine selbst geschriebene Machine-Learning Bibliothek zum Erstellen und Trainieren von KNNs (künstlichen Neuronalen Netzen). Es umfasst das Erstellen von neuronalen Netzen mit und ohne Hidden-Layer, verschiedene Aktivierungs-, Verlust- und Optimierungsfunktionen, einstellbare Trainingsparameter, Visualisierungsmethoden der KNNs und die Möglichkeit, Dateien einzulesen.

## Features

- Feedforward Neural Networks mit flexibler Schichtstruktur
- Aktivierungsfunktionen: ReLU, Swish, Sigmoid, Tanh, Snake, Softplus, Sinus, Linear
- Verlustfunktionen: MSE, MAE, Binary Cross-Entropy
- Optimierung: Gradient Descent mit Backpropagation
- Early Stopping, konfigurierbares Mini-Batching, Multi-Threading
- Modell-Export/Import (binäres Format)
- Visualisierung als Heatmap und Sankey-Plot
- 265 Testfälle mit 100 % Branch & Line Coverage

## Schnellstart
```bash
# JAR ins lokale Maven-Repository installieren
mvn install:install-file -Dfile=knn.jar -DpomFile=pom.xml
```
```xml
<dependency>
    <groupId>de.fhdw</groupId>
    <artifactId>knn</artifactId>
    <version>1.0.0</version>
</dependency>
```

Oder direkt per TOML-Konfiguration:
```bash
java -jar knn.jar example.toml
```

## Lizenz

Veröffentlicht unter der [ISC Lizenz](https://github.com/IP-TeamC/KNN/blob/main/LICENSE).  
Copyright © 2026 Marcel Anker, Lennart Heinrich, Piet Ostendorp