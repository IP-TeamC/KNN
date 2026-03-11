package de.fhdw.knn.config.network;

import de.fhdw.knn.config.Config;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.layer.DenseLayer;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;
import lombok.SneakyThrows;

/**
 * Die Klasse {@code ConfLayer} stellt eine Konfiguration für eine {@code DenseLayer} in einem neuronalen Netzwerk dar.
 * <p>Es umfasst Eigenschaften wie die Anzahl der Neuronen und die für die Schicht zu verwendende Aktivierungsfunktion.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 *
 * @see DenseLayer
 * @see TomlSerializable
 */
@Data
public class ConfLayer implements TomlSerializable {

    /**
     * Stellt die Anzahl der Neuronen in einer bestimmten {@code DenseLayer} dar.
     *
     * @see ConfLayer#create()
     * @see DenseLayer
     */
    private int neurons;
    /**
     * Stellt die Aktivierungsfunktion dar, die für eine {@code DenseLayer} innerhalb verwendet werden soll.
     *
     * @see ConfLayer#create()
     * @see DenseLayer#withActivationFunction(ActivationFunction)
     * @see ActivationFunction
     */
    private String activationFunction;


    /**
     * Erstellt ein neues {@code DenseLayer}-Objekt basierend auf der Konfiguration.
     * Die erstellte Schicht verfügt über die angegebene Anzahl von Neuronen und
     * verwendet die aus der Klassenkonfiguration abgerufene Aktivierungsfunktion.
     *
     * @return Eine {@code DenseLayer}-Instanz, die mit der angegebenen Anzahl von Neuronen und Aktivierungsfunktion konfiguriert ist.
     *
     * @see DenseLayer
     * @see ActivationFunction
     */
    @SneakyThrows
    public DenseLayer create() {
        ActivationFunction activationFunction = Config.getStaticField(ActivationFunction.class, getActivationFunction());
        return new DenseLayer(getNeurons()).withActivationFunction(activationFunction);
    }
}
