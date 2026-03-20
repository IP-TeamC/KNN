package de.fhdw.knn.data;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Ermöglicht mit der readFile-Methode das Einlesen von CSV-Dateien
 */
public class CsvReader {

    /**
     * Objekte dieser Klasse müssen nicht instanziiert werden. Diese Klasse besitzt nur static-Funktionen
     */
    public CsvReader() {
    }

    /**
     * Liest die CSV-Datei ein, ohne Zeilen zu überspringen
     *
     * @param fileName    Dateipfad zur einzulesenden Datei
     * @param inputStart  Startindex der Eingabedaten (Spalte)
     * @param inputSize   Größe der Eingabedaten (Zeile)
     * @param outputStart Startindex der Ausgabedaten (Spalte)
     * @param outputSize  Größe der Ausgabedaten (Zeile)
     * @return Dataset, eingeteilt in Eingabe- und Ausgabedaten, wie durch die Parameter definiert
     * @throws IOException Wird bei fehlern des BufferedReaders geworfen
     * @see CsvReader#readFile(String, int, int, int, int, int)
     */
    public static DataSet readFile(String fileName, int inputStart, int inputSize, int outputStart, int outputSize) throws IOException {
        return readFile(fileName, inputStart, inputSize, outputStart, outputSize, 0);
    }

    /**
     * Liest die CSV-Datei ein und überspringt dabei ersten angegebenen Zeilen
     *
     * @param fileName    Dateipfad zur einzulesenden Datei
     * @param inputStart  Startindex der Eingabedaten (Spalte)
     * @param inputSize   Größe der Eingabedaten (Zeile)
     * @param outputStart Startindex der Ausgabedaten (Spalte)
     * @param outputSize  Größe der Ausgabedaten (Zeile)
     * @param skip        Überspringt die ersten skip Zeilen; startet sonst in Header Zeile
     * @return Dataset, eingeteilt in Eingabe- und Ausgabedaten, wie durch die Parameter definiert
     * @throws IOException Wird bei fehlern des BufferedReaders geworfen
     */
    public static DataSet readFile(String fileName, int inputStart, int inputSize, int outputStart, int outputSize, int skip) throws IOException {
        return readFile(fileName, skip, line -> {
                    double[] dataset = Arrays.stream(line.split(",")).mapToDouble(CsvReader::parseDoubleOrNaN).toArray();
                    double[] input = new double[inputSize];
                    double[] output = new double[outputSize];
                    System.arraycopy(dataset, inputStart, input, 0, inputSize);
                    System.arraycopy(dataset, outputStart, output, 0, outputSize);
                    return Stream.of(Map.entry(input, output));
                },
                (dataSet, labels) -> {
                    dataSet.inputLabels = new String[inputSize];
                    dataSet.outputLabels = new String[outputSize];

                    String[] headers = labels.split(",");
                    System.arraycopy(headers, inputStart, dataSet.inputLabels, 0, inputSize);
                    System.arraycopy(headers, outputStart, dataSet.outputLabels, 0, outputSize);
                });
    }

    /**
     * Liest die CSV-Datei ein und überspringt dabei ersten angegebenen Zeilen
     *
     * @param fileName   Dateipfad zur einzulesenden Datei
     * @param skip       Überspringt die ersten skip Zeilen; startet sonst in Header Zeile
     * @param lineParser Funktion, welche jede Zeile der eingelesenen Datei in ein Paar/Entry aus Eingabe- und Ausgabe-Array konvertiert
     * @return Dataset, eingeteilt in Eingabe- und Ausgabedaten, wie durch die Parameter definiert
     * @throws IOException Wird bei fehlern des BufferedReaders geworfen
     */
    public static DataSet readFile(String fileName, int skip, Function<String, Stream<Map.Entry<double[], double[]>>> lineParser) throws IOException {
        return readFile(fileName, skip, lineParser, null);
    }

    /**
     * Liest die CSV-Datei ein und überspringt dabei ersten angegebenen Zeilen
     *
     * @param fileName    Dateipfad zur einzulesenden Datei
     * @param skip        Überspringt die ersten skip Zeilen; startet sonst in Header Zeile
     * @param lineParser  Funktion, welche jede Zeile der eingelesenen Datei in ein Paar/Entry aus Eingabe- und Ausgabe-Array konvertiert
     * @param labelParser Verarbeitet die Header-Zeile (erste Zeile, wenn skip > 0) und kann das DataSet dabei anpassen. Akzeptiert Dataset, String
     * @return Dataset, eingeteilt in Eingabe- und Ausgabedaten, wie durch die Parameter definiert
     * @throws java.io.IOException Bei einem Fehler mit dem BufferedReader
     */
    public static DataSet readFile(String fileName, int skip,
                                   Function<String, Stream<Map.Entry<double[], double[]>>> lineParser,
                                   BiConsumer<DataSet, String> labelParser) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String labels = null;
            if (skip > 0 && labelParser != null) {
                labels = br.readLine();
                skip -= 1;
            }
            List<Map.Entry<double[], double[]>> data = br.lines().skip(skip).flatMap(lineParser).toList();

            if (data.isEmpty()) {
                throw new IllegalArgumentException("Csv-File not allowed to be empty.");
            }

            double[][] input = new double[data.size()][];
            double[][] output = new double[data.size()][];
            for (int i = 0; i < data.size(); i++) {
                input[i] = data.get(i).getKey();
                output[i] = data.get(i).getValue();
            }

            DataSet dataSet = new DataSet(input, output);
            if (labels != null) {
                labelParser.accept(dataSet, labels);
            }

            return dataSet;
        }
    }


    /**
     * Initialisiert, wenn möglich, einen Double aus dem String-Value.
     *
     * @param value String, welcher zu einem Double umgewandelt wird
     * @return Double, wenn der String nicht umgewandelt werden kann, Double.NaN
     */
    private static double parseDoubleOrNaN(String value) {
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }

}
