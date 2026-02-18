package de.fhdw.knn.network;

import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.io.Exporter;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.OutputsDerived;
import de.fhdw.knn.network.neuron.OutputDerived;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * Fasst die gesamte Netzstruktur eines neuronalen Netzwerks zusammen.
 * Besteht aus einem Input Layer, Hidden Layer (optional) und einem Output Layer
 */
public class Network {

    public final Random random;

    public InputLayer inputLayer;
    public DenseLayer[] denseLayers;

    /**
     * Erzeugt ein Netzwerk aus den gegebenen Layern.
     * <strong>Ein hierüber erzeugtes Netz kann keine Verbindungen/Gewichte initialisieren (kein Seed gesetzt).</strong>
     */
    public Network(InputLayer inputLayer, DenseLayer[] denseLayers) {
        this.random = null;
        this.inputLayer = inputLayer;
        this.denseLayers = denseLayers;
    }

    /**
     * Erzeugt ein Netzwerk aus den gegebenen Layern.
     * Es werden alle Neuronen wie in einem Feedforward-KNN üblich verbunden und die Verbindungen mit Gewichten entsprechend initialisiert.
     */
    public Network(long seed, WeightInitializer weightInitializer, InputLayer inputLayer, DenseLayer... denseLayers) {
        this.random = new Random(seed);
        this.inputLayer = inputLayer;
        this.denseLayers = denseLayers;
        this.connectAll(weightInitializer);
    }

    /**
     * Erzeugt ein Netzwerk aus den gegebenen Dense Layern.
     * Der Input Layer wird mit der gewünschten Anzahl an Input-Neuronen neu erzeugt.
     * Es werden alle Neuronen wie in einem Feedforward-KNN üblich verbunden und die Verbindungen mit Gewichten entsprechend initialisiert.
     */
    public Network(long seed, WeightInitializer weightInitializer, int inputNeurons, DenseLayer... denseLayers) {
        this.random = new Random(seed);
        this.inputLayer = new InputLayer(inputNeurons);
        this.denseLayers = denseLayers;
        this.connectAll(weightInitializer);
    }

    /**
     * Ermittelt für mehrere Zeilen/Eingaben alle Ausgabe-Zeilen bei Verwendung des Netzwerks
     * @param inputs außen Eingabe-Zeile - innen Spalte/Feature/Merkmal (Ausgabe des Input-Neurons)
     * @return außen Ausgabe-Zeile - innen Spalte/Feature/Merkmal (Ausgabe des Output-Neurons)
     */
    public double[][] predict(double[][] inputs) {
        double[][] predictions = new double[inputs.length][];
        IntStream.range(0, inputs.length).parallel().forEach(i -> predictions[i] = feedForward(inputs[i]).lastOutput());
        return predictions;
    }

    /**
     * Ermittelt für mehrere Zeilen/Eingaben alle Ausgabe-Zeilen bei Verwendung des Netzwerks.
     * Es wird jedoch nur das erste/einzige Output-Neuron beachtet.
     * @param inputs außen Eingabe-Zeile - innen Spalte/Feature/Merkmal (Ausgabe des Input-Neurons)
     * @return Ausgabe-Zeilen mit je nur einem Output-Neuron
     */
    public double[] predictSingles(double[][] inputs) {
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = feedForward(inputs[i]).lastOutput()[0];
        }
        return predictions;
    }

    @Deprecated
    public double[] predictSingles(double[] inputs) {
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = feedForward(new double[]{inputs[i]}).lastOutput()[0];
        }
        return predictions;
    }

    /**
     * Berechnet die Ausgaben/Aktivierungen aller DenseNeuronen sowie deren Ableitungen bei Eingabe einer Zeile (Feed-Forward).
     * @param input eine Eingabe-Zeile
     * @return Ausgaben/Aktivierungen aller DenseNeuronen sowie deren Ableitungen (außen Layer - innen Neuron im Layer)
     */
    public OutputsDerived feedForward(double[] input) {
        double[][] output = new double[denseLayers.length][];
        double[][] derived = new double[denseLayers.length][];

        calculateOutput(0, input, output, derived);
        for (int layer = 1; layer < denseLayers.length; layer++) {
            calculateOutput(layer, output[layer - 1], output, derived);
        }

        return new OutputsDerived(output, derived);
    }

    /**
     * Berechnet die Ausgaben/Aktivierungen aller DenseNeuronen sowie deren Ableitungen innerhalb eines Layers auf Basis der Aktivierungen des vorherigen Layers
     */
    private void calculateOutput(int layer, double[] input, double[][] output, double[][] derived) {
        AbstractDenseNeuron[] neurons = denseLayers[layer].neurons;
        output[layer] = new double[neurons.length];
        derived[layer] = new double[neurons.length];

        for (int neuron = 0; neuron < neurons.length; neuron++) {
            AbstractDenseNeuron dn = denseLayers[layer].neurons[neuron];
            OutputDerived neuronOutput = dn.compute(input);
            output[layer][neuron] = neuronOutput.output();
            derived[layer][neuron] = neuronOutput.derived();
        }
    }

    /**
     * Es werden alle Neuronen wie in einem Feedforward-KNN üblich verbunden und die Verbindungen mit Gewichten entsprechend initialisiert.
     */
    private void connectAll(WeightInitializer weightInitializer) {
        for (AbstractDenseNeuron denseNeuron : denseLayers[0].neurons) {
            if (denseNeuron.incoming != null) continue;
            denseNeuron.incoming = new Connection[inputLayer.neurons.length];

            for (int inputNeuron = 0; inputNeuron < inputLayer.neurons.length; inputNeuron++) {
                Connection conn = new Connection(weightInitializer.nextWeight(random, this, 0));
                conn.inputNeuron = inputLayer.neurons[inputNeuron];
                denseNeuron.incoming[inputNeuron] = conn;
            }
        }

        for (int layer = 1; layer < denseLayers.length; layer++) {
            AbstractDenseNeuron[] prevNeurons = denseLayers[layer - 1].neurons;
            AbstractDenseNeuron[] currentNeurons = denseLayers[layer].neurons;

            for (AbstractDenseNeuron currentNeuron : currentNeurons) {
                if (currentNeuron.incoming != null) continue;
                currentNeuron.incoming = new Connection[prevNeurons.length];

                for (int prevNeuron = 0; prevNeuron < prevNeurons.length; prevNeuron++) {
                    Connection conn = new Connection(weightInitializer.nextWeight(random, this, layer));
                    conn.inputNeuron = prevNeurons[prevNeuron];
                    currentNeuron.incoming[prevNeuron] = conn;
                }
            }
        }
    }

    /**
     * Exportiert das neuronale Netz in eine Datei
     * @param fileName Dateipfad
     */
    public void export(String fileName) {
        Exporter.export(this, fileName);
    }

}
