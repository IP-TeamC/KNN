package de.fhdw.knn.config.network;

import de.fhdw.knn.config.Config;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.io.Importer;
import de.fhdw.knn.network.layer.DenseLayer;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

import java.util.Arrays;
import java.util.Optional;

/**
 * Die Klasse {@code ConfNetwork} stellt eine Konfiguration für eine neuronales Netzwerk dar.
 * <p>Diese Klasse bietet Funktionen zum Initialisieren eines Netzwerks basierend auf seinen
 * Konfigurationseigenschaften oder zum Importieren eines bestehenden Netzwerks aus einer Datei.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 *
 * @see Network
 * @see TomlSerializable
 */
@Data
public class ConfNetwork implements TomlSerializable {

    /**
     * Stellt den Pfad zu einer externen Datei dar, aus der ein vorkonfiguriertes neuronales Netzwerk
     * importiert werden kann.
     *
     * @see ConfNetwork#create(DataSet)
     */
    private String importFile;
    /**
     * Stellt den Zufallswert dar, der zur Initialisierung der Zufälligkeit in der Konfiguration
     * des neuronalen Netzwerks verwendet wird.
     *
     * @see ConfNetwork#create(DataSet)
     */
    private int seed;
    /**
     * Gibt den Gewichtsinitialisierer an, der zum Initialisieren der Gewichte
     * des neuronalen Netzwerks verwendet werden soll.
     *
     * @see WeightInitializer
     * @see ConfNetwork#create(DataSet)
     */
    private String weightInitializer;
    /**
     * Repräsentiert die Konfiguration der Schichten eines neuronalen Netzwerks.
     *
     * @see ConfLayer
     * @see ConfNetwork#create(DataSet)
     */
    private ConfLayer[] layer;

    /**
     * Erstellt eine neuronale Netzwerkinstanz basierend auf der aktuellen Konfiguration oder
     * importiert eine vorkonfigurierte Instanz aus einer Datei.
     *
     * <p>Wenn {@code importFile} angegeben ist, können zusätzliche Konfigurationseigenschaften wie
     * {@code seed}, {@code weightInitializer} oder {@code layer} nicht verwendet werden und führen zu einer Exception.
     *
     * @param data Das {@code DataSet}, das Informationen zur Eingabegröße enthält, die in der Netzwerkkonfiguration verwendet werden.
     * @return Eine {@code Network}-Instanz, die gemäß den angegebenen Eigenschaften konfiguriert oder aus der Datei importiert wurde.
     * @throws IllegalArgumentException Wenn sowohl {@code importFile} angegeben als auch zusätzliche Konfigurationseigenschaften
     *                                  ({@code seed}, {@code weightInitializer} oder {@code layer}) festgelegt sind.
     *
     * @see Network
     */
    public Network create(DataSet data) {
        if (importFile != null) {
            if (seed != 0 || weightInitializer != null || layer != null) {
                throw new IllegalArgumentException("No network config allowed when using importFile");
            }
            return Importer.importNetwork(importFile);
        }

        DenseLayer[] denseLayers = Arrays.stream(layer)
                .map(ConfLayer::create)
                .toArray(DenseLayer[]::new);
        WeightInitializer weightInitializer = Optional.ofNullable(this.weightInitializer)
                .map(value -> Config.getStaticField(WeightInitializer.class, value))
                .orElse(WeightInitializer.GLOROT_UNIFORM);
        return new Network(seed, weightInitializer, data.inputSize, denseLayers);
    }
}
