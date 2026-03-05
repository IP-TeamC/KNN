package de.fhdw.knn.network.io;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.SuperNeuron;
import lombok.SneakyThrows;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Consumer;

/**
 * Importer zum Erzeugen eines {@link Network}-Objekts aus einem exportierten Netzwerk (KNN-Datei)
 */
public class Importer {

    /**
     * Verwendung des State-Pattern zum Deserialisieren einzelner Abschnitte des exportierten Netzwerks
     */
    private Consumer<ByteBuffer> state = this::createInputLayer;

    /**
     * Deserialisierter Input-Layer
     */
    private InputLayer inputLayer;
    /**
     * Deserialisierte Dense-Layer
     */
    private DenseLayer[] denseLayers;
    /**
     * Gesamtes deserialisiertes Netzwerk
     */
    private Network network;

    /**
     * Position des aktuell betrachteten Dense Layers
     */
    private int denseLayer = 0;
    /**
     * Position des aktuell betrachteten Neurons
     */
    private int neuron = 0;
    /**
     * Position der aktuell betrachteten Verbindung
     */
    private int conn = 0;

    /**
     * Importiert das Netzwerk aus dem übergebenen KNN-Datei-Pfad
     *
     * @param fileName Datei-Pfad des zu importierenden Netzwerks
     * @return Importiertes Netzwerk
     */
    public static Network importNetwork(String fileName) {
        return new Importer().load(fileName);
    }

    /**
     * Importiert das Netzwerk aus dem übergebenen Byte-Array<br>
     * <strong>WICHTIG! Diese Methode darf für das gleiche Objekt nur einmal aufgerufen werden.
     * Jeder Import muss mit einem weiteren {@link Importer}-Objekt passieren.</strong><br>
     * <p>
     * Das Format entspricht folgendem:<br><br>
     * Anzahl Input-Neurons<br>
     * Anzahl Dense Layer<br>
     * { je Layer<br>
     * Anzahl Neuronen<br>
     * }<br>
     * { je Neuron je Layer (0 -> n)<br>
     * ActivationFunction (Integer.MAX_VALUE = Super-Neuron)<br>
     * { wenn DenseNeuron<br>
     * BIAS<br>
     * }<br>
     * { sonst wenn SuperNeuron<br>
     * Network-Size<br>
     * Network<br>
     * }<br>
     * (Guard als byte, Weight)... je Connection (0 -> n)<br>
     * }
     */
    public Network load(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        while (buffer.position() != buffer.capacity()) {
            state.accept(buffer);
        }
        return network;
    }

    /**
     * Importiert das Netzwerk aus dem übergebenen KNN-Datei-Pfad
     * <strong>WICHTIG! Diese Methode darf für das gleiche Objekt nur einmal aufgerufen werden.
     * Jeder Import muss mit einem weiteren {@link Importer}-Objekt passieren.</strong>
     *
     * @param fileName Datei-Pfad des zu importierenden Netzwerks
     * @return Importiertes Netzwerk
     */
    @SneakyThrows
    public Network load(String fileName) {
        return load(Files.readAllBytes(Paths.get(fileName)));
    }

    /**
     * Liest die Größe des Input Layers und erzeugt diesen
     *
     * @param buffer ByteBuffer, das weiter eingelesen wird (als Stream)
     */
    private void createInputLayer(ByteBuffer buffer) {
        inputLayer = new InputLayer(buffer.getInt());
        state = this::createDenseLayers;
    }

    /**
     * Liest die Anzahl der Dense Layers und erzeugt das Array für diese
     *
     * @param buffer ByteBuffer, das weiter eingelesen wird (als Stream)
     */
    private void createDenseLayers(ByteBuffer buffer) {
        denseLayers = new DenseLayer[buffer.getInt()];
        state = this::createDenseLayer;
    }

    /**
     * Liest die Größe eines weiteren Dense Layers und erzeugt diesen
     *
     * @param buffer ByteBuffer, das weiter eingelesen wird (als Stream)
     */
    private void createDenseLayer(ByteBuffer buffer) {
        denseLayers[denseLayer++] = new DenseLayer(buffer.getInt());
        if (denseLayer >= denseLayers.length) {
            network = new Network(0, WeightInitializer.ZERO, inputLayer, denseLayers);
            denseLayer = 0;
            state = this::updateActivationFunctionAndBias;
        }
    }

    /**
     * Liest die Aktivierungsfunktion und den Bias eines DenseNeurons und erzeugt dieses Neuron
     *
     * @param buffer ByteBuffer, das weiter eingelesen wird (als Stream)
     */
    private void updateActivationFunctionAndBias(ByteBuffer buffer) {
        int activationFunction = buffer.getInt();
        if (activationFunction == Integer.MAX_VALUE) {
            state = this::updateSuperNeuron;
            return;
        }

        DenseNeuron dn = (DenseNeuron) denseLayers[denseLayer].neurons[neuron];
        dn.activationFunction = ActivationFunction.FUNCTIONS.get(activationFunction);
        dn.bias = buffer.getDouble();
        state = this::updateWeights;
    }

    /**
     * Liest das Netzwerk eines SuperNeurons ein und erzeugt das SuperNeuron
     *
     * @param buffer ByteBuffer, das weiter eingelesen wird (als Stream)
     */
    private void updateSuperNeuron(ByteBuffer buffer) {
        int subnetSize = buffer.getInt();
        byte[] subnetRaw = new byte[subnetSize];
        buffer.get(subnetRaw);
        Network subnet = new Importer().load(subnetRaw);

        new SuperNeuron(subnet).insert(network, denseLayer, neuron);
        state = this::updateWeights;
    }

    /**
     * Liest die den Guard und das Gewicht einer Verbindung ein und setzt diese
     *
     * @param buffer ByteBuffer, das weiter eingelesen wird (als Stream)
     */
    private void updateWeights(ByteBuffer buffer) {
        denseLayers[denseLayer].neurons[neuron].incoming[conn].guard = buffer.get() >= 1;
        denseLayers[denseLayer].neurons[neuron].incoming[conn++].weight = buffer.getDouble();
        if (conn >= denseLayers[denseLayer].neurons[neuron].incoming.length) {
            neuron += 1;
            conn = 0;
            state = this::updateActivationFunctionAndBias;

            if (neuron >= denseLayers[denseLayer].neurons.length) {
                denseLayer += 1;
                neuron = 0;
            }
        }
    }

}
