package de.fhdw.knn.config.data;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.TrainTestSplit;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

import java.io.IOException;

/**
 * Stellt die Konfigurationsdaten für die Eingabe- und Ausgabeparameter dar, die zur Verarbeitung
 * von CSV-Dateien und zur Erstellung von Trainings- und Testdatensätzen verwendet werden.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 *
 * @see TomlSerializable
 */
@Data
public class ConfData implements TomlSerializable {

    /**
     * Der Pfad oder Dateiname, der auf die zu verarbeitende CSV-Datei verweist.
     *
     * <p>Diese Datei dient als Eingabequelle für Trainings- und Testdatensätze,
     * die durch die Methode {@link ConfData#create()} verarbeitet werden.
     *
     * <p>Es wird erwartet, dass die Datei im CSV-Format vorliegt.
     */
    private String file;
    /**
     * Gibt den Index der Spalte an, ab der die Eingabedaten im CSV-Datensatz beginnen.
     *
     * @see #inputSize
     */
    private int inputStart;
    /**
     * Definiert die Anzahl der Eingabemerkmale (Features), die aus den CSV-Daten extrahiert werden sollen.
     * Sie wird in Kombination mit {@code inputStart} verwendet, um den Umfang der Eingabedaten zu definieren.
     *
     * @see #inputStart
     */
    private int inputSize;
    /**
     * Gibt den Index der Spalte an, ab der die Ausgabedaten im CSV-Datensatz beginnen.
     *
     * @see #outputSize
     */
    private int outputStart;
    /**
     * Definiert die Anzahl der Ausgabemerkmale (Features), die aus den CSV-Daten extrahiert werden sollen.
     * Sie wird in Kombination mit {@code outputStart} verwendet, um den Umfang der Ausgabedaten zu definieren.
     *
     * @see #outputStart
     */
    private int outputSize;
    /**
     * Definiert die Anzahl der Zeilen, die beim Einlesen der CSV-Datei übersprungen werden sollen.
     *
     * <p>Diese Variablenkonfiguration wird von der Methode {@link CsvReader#readFile(String, int, int, int, int, int)}
     * verwendet, um zu bestimmen, wie viele Zeilen am Anfang der Datei ignoriert werden. Sie ermöglicht es,
     * Headerzeilen oder andere nicht relevante Teilabschnitte der Datei auszuschließen.
     *
     * @see CsvReader#readFile(String, int, int, int, int, int)
     */
    private int skip;

    /**
     * Der Zufallswert, der zum Mischen und Aufteilen der Daten bei der Erstellung
     * der Trainings- und Testdatensätze verwendet wird.
     *
     * <p>Es wird im {@link TrainTestSplit}-Erstellungsprozess verwendet,
     * der in der Methode {@link ConfData#create()} implementiert ist.
     */
    private int seed;
    /**
     * Gibt den Anteil der Daten an, der für den Testdatensatz bei der Aufteilung des
     * vollständigen Datensatzes in Trainings- und Testuntergruppen zugewiesen werden soll.
     *
     * <p>Ein Wert zwischen 0 und 1, wobei:
     * <ul>
     *      <li>Ein Wert von 0 bedeutet, dass dem Testdatensatz keine Daten zugewiesen werden
     *          und alle Daten im Trainingsdatensatz verbleiben.</li>
     *      <li>Ein Wert von 1 bedeutet, dass alle Daten dem Testdatensatz zugewiesen werden
     *          und keine Daten im Trainingsdatensatz verbleiben.</li>
     * </ul>
     */
    private double testShare;


    /**
     * Erstellt einen Trainings- und Testdatensatz aus den Konfigurationsdaten.
     *
     * @return ein {@code TrainTestSplit} Objekt, dass die Trainings- und Testdatensätze enthält.
     * @throws IOException wenn beim Lesen der CSV-Datei ein Fehler auftritt.
     *
     * @see TrainTestSplit
     */
    public TrainTestSplit create() throws IOException {
        return CsvReader.readFile(file, inputStart, inputSize, outputStart, outputSize, skip).shuffleAndSplit(seed, testShare);
    }
}
