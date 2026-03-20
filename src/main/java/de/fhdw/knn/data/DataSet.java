package de.fhdw.knn.data;

import java.util.Random;
import java.util.function.BiConsumer;

/**
 * Kapselt einen Datensatz mit Eingabe-/Ausgabe-Daten, Labels, Größe und Normalisierung.
 */
public class DataSet {

    /**
     * Anzahl Zeilen im Datensatz
     */
    public final int size;
    /**
     * Anzahl Spalten in der Eingabe
     */
    public final int inputSize;
    /**
     * Anzahl Spalten in der Ausgabe
     */
    public final int outputSize;

    /**
     * Eingabe-Daten: außen Zeile - innen Spalte
     */
    public double[][] inputs;
    /**
     * Ausgabe-Daten: außen Zeile - innen Spalte
     */
    public double[][] outputs;

    /**
     * Eingabe-Label je Spalte
     */
    public String[] inputLabels;
    /**
     * Ausgabe-Label je Spalte
     */
    public String[] outputLabels;

    /**
     * Gespeicherter Normalizer der Eingaben, um diese ggf. wieder denormalisieren zu können
     */
    private Normalizer normalizerInputs;

    /**
     * Gespeicherter Normalizer der Ausgaben, um diese ggf. wieder denormalisieren zu können
     */
    private Normalizer normalizerOutputs;

    /**
     * Erzeugt aus dem Eingabe-/Ausgabe-Array einen Datensatz.
     *
     * @param inputs  enthält alle Eingabe-Zeilen (inneres Array entspricht einer Zeile)
     * @param outputs enthält alle Ausgabe-Zeilen (inneres Array entspricht einer Zeile)
     * @see DataSet
     */
    public DataSet(double[][] inputs, double[][] outputs) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("input.length != output.length");
        }

        this.size = inputs.length;
        this.inputSize = inputs[0].length;
        this.outputSize = outputs[0].length;

        this.inputs = inputs;
        this.outputs = outputs;
    }

    /**
     * Führt den Preprocessor für jede Zeile (Eingabe/Ausgabe) des DataSets aus
     *
     * @param preprocessor Verarbeitet jede Zeile vorher. Akzeptiert Eingabe- und Ausgabedaten
     */
    public void preprocess(BiConsumer<double[], double[]> preprocessor) {
        for (int i = 0; i < size; i++) {
            preprocessor.accept(inputs[i], outputs[i]);
        }
    }

    /**
     * Erzeugt einen Datensatz, der in der Eingabe nur einen Teil der Spalten enthält.
     * Die Reihenfolge der Spalten entspricht der Reihenfolge der als Parameter übergebenen Spalten-Indizes.
     *
     * @param columns Beliebige Anzahl an Spalten-Indizes
     * @return Dataset, mit dem Subset als Eingabe und normalen Ausgaben
     */
    public DataSet subsetInputs(int... columns) {
        double[][] subset = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            subset[i] = new double[columns.length];
            int newColumn = 0;
            for (int column : columns) {
                subset[i][newColumn] = inputs[i][column];
                newColumn += 1;
            }
        }
        return new DataSet(subset, outputs);
    }

    /**
     * Mischt die Zeilen zufällig und verwendet für den RNG den übergebenen Seed (dadurch deterministisch).
     *
     * @param seed Verwendeter Random Seed für den shuffle
     */
    public void shuffle(long seed) {
        Random random = new Random(seed);
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] output = outputs[i];
            int pos = random.nextInt(inputs.length);

            inputs[i] = inputs[pos];
            outputs[i] = outputs[pos];

            inputs[pos] = input;
            outputs[pos] = output;
        }
    }

    /**
     * Mischt die Zeilen zufällig und verwendet für den RNG den übergebenen Seed (dadurch deterministisch).
     * Teilt danach den letzten Anteil der Daten (testShare zwischen 0 und 1) den Test-Daten zu.
     * Der vordere Teil wird als Trainings-Daten verwendet.
     *
     * @param seed      Verwendeter Random Seed für den shuffle
     * @param testShare Prozentualer Anzeil an Daten in dezimalschreibweise, die in den Test-Anteil sollen
     * @return TrainTestSplit, der jeweils aus einem Test-Dataset und einem Train-Dataset besteht
     * @see DataSet#shuffle(long)
     */
    public TrainTestSplit shuffleAndSplit(long seed, double testShare) throws IllegalArgumentException {
        shuffle(seed);

        if (testShare < 0 || testShare > 1) {
            throw new IllegalArgumentException("Test share must be between 0 and 1");
        }

        int testSize = (int) (testShare * size);
        int trainSize = size - testSize;

        DataSet train = null;
        double[][] inputsTrain = new double[trainSize][];
        double[][] outputsTrain = new double[trainSize][];
        System.arraycopy(inputs, 0, inputsTrain, 0, trainSize);
        System.arraycopy(outputs, 0, outputsTrain, 0, trainSize);
        if (testShare < 1) {
            train = new DataSet(inputsTrain, outputsTrain);
        }

        DataSet test = null;
        double[][] inputsTest = new double[testSize][];
        double[][] outputsTest = new double[testSize][];
        System.arraycopy(inputs, trainSize, inputsTest, 0, testSize);
        System.arraycopy(outputs, trainSize, outputsTest, 0, testSize);
        if (testShare > 0) {
            test = new DataSet(inputsTest, outputsTest);
        }

        return new TrainTestSplit(train, test);
    }

    /**
     * Normalisiert alle Eingabe-Daten mit dem Normalizer und speichert diesen, um eine spätere Denormalisierung zu ermöglichen.
     *
     * @param normalizer Spezifischer normalizer für Eingaben
     */
    public void normalizeInputs(Normalizer normalizer) {
        normalizerInputs = normalizer;
        normalizerInputs.normalize(inputs);
    }

    /**
     * Normalisiert alle Ausgabe-Daten mit dem Normalizer und speichert diesen, um eine spätere Denormalisierung zu ermöglichen.
     *
     * @param normalizer Spezifischer normalizer für Ausgaben
     */
    public void normalizeOutputs(Normalizer normalizer) {
        normalizerOutputs = normalizer;
        normalizerOutputs.normalize(outputs);
    }

    /**
     * Denormalisiert alle Eingabe-Daten mit dem Normalizer, der zuvor beim Aufruf von {@link DataSet#normalizeInputs(Normalizer)} übergeben wurde.
     */
    public void denormalizeInputs() {
        normalizerInputs.denormalize(inputs);
    }

    /**
     * Denormalisiert alle Ausgabe-Daten mit dem Normalizer, der zuvor beim Aufruf von {@link DataSet#normalizeOutputs(Normalizer)} übergeben wurde.
     */
    public void denormalizeOutputs() {
        normalizerOutputs.denormalize(outputs);
    }

}
