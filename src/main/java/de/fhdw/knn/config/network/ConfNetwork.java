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

@Data
public class ConfNetwork implements TomlSerializable {

    private String importFile;
    private int seed;
    private String weightInitializer;
    private ConfLayer[] layer;

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
