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

    /**
     * RNG für die deterministische Erzeugung der Gewichte und des Bias der Neuronen
     */
    public final Random random;

    /**
     * Input Layer des Netzwerks
     */
    public InputLayer inputLayer;
    /**
     * Hidden Layer und Output Layer des Netzwerks
     */
    public DenseLayer[] denseLayers;

    /**
     * Erzeugt ein Netzwerk aus den gegebenen Layern.
     * <strong>Ein hierüber erzeugtes Netz kann keine Verbindungen/Gewichte initialisieren (kein Seed gesetzt).</strong>
     *
     * @param inputLayer  Input-Layer des Netzwerks
     * @param denseLayers Dense-Layer des Netzwerks (mehrere Hidden Layer und Output Layer als letzter Dense Layer)
     */
    public Network(InputLayer inputLayer, DenseLayer[] denseLayers) {
        this.random = null;
        this.inputLayer = inputLayer;
        this.denseLayers = denseLayers;
    }

    /**
     * Erzeugt ein Netzwerk aus den gegebenen Layern.
     * Es werden alle Neuronen wie in einem Feedforward-KNN üblich verbunden und die Verbindungen mit Gewichten entsprechend initialisiert.
     *
     * @param seed              Seed zur Initialisierung des RNG für die Gewichtsinitialisierung
     * @param weightInitializer Algorithmus/Methode zur Gewichtsinitialisierung
     * @param inputLayer        Input-Layer des Netzwerks
     * @param denseLayers       Dense-Layer des Netzwerks (mehrere Hidden Layer und Output Layer als letzter Dense Layer)
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
     *
     * @param seed              Seed zur Initialisierung des RNG für die Gewichtsinitialisierung
     * @param weightInitializer Algorithmus/Methode zur Gewichtsinitialisierung
     * @param inputNeurons      Anzahl der gewünschten Neuronen im Input Layer (wird automatisch bei diesem Konstruktor erzeugt)
     * @param denseLayers       Dense-Layer des Netzwerks (mehrere Hidden Layer und Output Layer als letzter Dense Layer)
     */
    public Network(long seed, WeightInitializer weightInitializer, int inputNeurons, DenseLayer... denseLayers) {
        this.random = new Random(seed);
        this.inputLayer = new InputLayer(inputNeurons);
        this.denseLayers = denseLayers;
        this.connectAll(weightInitializer);
    }

    /**
     * Ermittelt für mehrere Zeilen/Eingaben alle Ausgabe-Zeilen bei Verwendung des Netzwerks
     *
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
     * Es wird jedoch nur das erste Output-Neuron beachtet (für Netzwerke mit nur einem Output-Neuron).
     *
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

    /**
     * Berechnet die Ausgaben/Aktivierungen aller DenseNeuronen sowie deren Ableitungen bei Eingabe einer Zeile (Feed-Forward).
     * Verwendung ohne Buffer siehe {@link Network#feedForward(double[])}.
     *
     * @param buffer wird mit den Ausgaben/Aktivierungen und deren Ableitungen aller Neuronen überschrieben
     * @param input eine Eingabe-Zeile
     */
    public void feedForward(OutputsDerived buffer, double[] input) {
        calculateOutput(0, input, buffer.output[0], buffer.derived[0]);
        for (int layer = 1; layer < denseLayers.length; layer++) {
            calculateOutput(layer, buffer.output[layer - 1], buffer.output[layer], buffer.derived[layer]);
        }
    }

    /**
     * Berechnet die Ausgaben/Aktivierungen aller DenseNeuronen sowie deren Ableitungen bei Eingabe einer Zeile (Feed-Forward).
     *
     * @param input eine Eingabe-Zeile
     * @return Ausgaben/Aktivierungen aller DenseNeuronen sowie deren Ableitungen (außen Layer - innen Neuron im Layer)
     */
    public OutputsDerived feedForward(double[] input) {
        OutputsDerived buffer = OutputsDerived.generateEmpty(this);
        feedForward(buffer, input);
        return buffer;
    }

    /**
     * Berechnet die Ausgaben/Aktivierungen aller DenseNeuronen sowie deren Ableitungen innerhalb eines Layers auf Basis der Aktivierungen des vorherigen Layers
     *
     * @param layer        Index des DenseLayers
     * @param input        Ausgabe des vorherigen Layers bzw. Eingabe in den Input Layer (bei layer = 0)
     * @param layerOutput  Array mit den Ausgaben aller Neuronen des Layers (wird überschrieben)
     * @param layerDerived Array mit den Ableitungen der Ausgabe/Aktivierung aller Neuronen des Layers (wird überschrieben)
     */
    private void calculateOutput(int layer, double[] input, double[] layerOutput, double[] layerDerived) {
        AbstractDenseNeuron[] neurons = denseLayers[layer].neurons;

        for (int neuron = 0; neuron < neurons.length; neuron++) {
            AbstractDenseNeuron dn = denseLayers[layer].neurons[neuron];
            OutputDerived neuronOutput = dn.compute(input);
            layerOutput[neuron] = neuronOutput.output();
            layerDerived[neuron] = neuronOutput.derived();
        }
    }

    /**
     * Es werden alle Neuronen wie in einem Feedforward-KNN üblich verbunden und die Verbindungen mit Gewichten entsprechend initialisiert.
     * Jedes Neuron wird mit allen Neuronen des vorherigen Layers verbunden.
     *
     * @param weightInitializer Algorithmus/Methode zur Gewichtsinitialisierung (wird für alle Neuronen vom vordersten Dense Layer zum Output Layer und vom geringsten Neuronen-Index zum größten Neuronen-Index aufgerufen)
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
     *
     * @param fileName Dateipfad
     */
    public void export(String fileName) {
        Exporter.export(this, fileName);
    }

}
