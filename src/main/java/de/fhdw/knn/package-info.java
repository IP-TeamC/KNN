/**
 * Dieses Modul stellt eine Bibliothek zur Erstellung und zum Training eines KNNs (künstliches neuronales Netzwerk) dar.<br>
 * Es besteht aus einem {@link de.fhdw.knn.data.CsvReader} zum Einlesen der Daten,<br>
 * einem {@link de.fhdw.knn.network.Network} zum Erzeugen der Netzstruktur,<br>
 * einem {@link de.fhdw.knn.trainer.Trainer} zum Trainieren des Modells,<br>
 * einem {@link de.fhdw.knn.scorer.Scorer} zum Evaluieren der Ergebnisse sowie<br>
 * einem {@link de.fhdw.knn.visualization.LiveViewManager} zum Darstellen der Netzstruktur.<br>
 * <br>
 * Alle relevanten Klassen, die hierfür zusätzlich verwendet werden müssen,
 * sowie alle öffentlichen Methoden und Attribute verfügen über eine Javadoc-Dokumentation.
 *
 * @version 2026.2.1
 * @author Marcel Anker
 * @author Lennart Heinrich
 * @author Piet Ostendorp
 */
package de.fhdw.knn;