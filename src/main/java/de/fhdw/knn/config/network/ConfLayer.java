package de.fhdw.knn.config.network;

import de.fhdw.knn.config.Config;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.layer.DenseLayer;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;
import lombok.SneakyThrows;

@Data
public class ConfLayer implements TomlSerializable {

    private int neurons;
    private String activationFunction;

    @SneakyThrows
    public DenseLayer create() {
        ActivationFunction activationFunction = Config.getStaticField(ActivationFunction.class, getActivationFunction());
        return new DenseLayer(getNeurons()).withActivationFunction(activationFunction);
    }

}
