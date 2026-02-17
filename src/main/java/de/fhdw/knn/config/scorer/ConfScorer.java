package de.fhdw.knn.config.scorer;

import de.fhdw.knn.config.Config;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.scorer.ClassificationScorer;
import de.fhdw.knn.scorer.Scorer;
import de.fhdw.knn.trainer.loss.LossFunction;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

import java.util.Optional;

@Data
public class ConfScorer implements TomlSerializable {

    private String lossFunction;
    private String type;

    public Optional<Scorer> create(Network network) {
        //noinspection SwitchStatementWithTooFewBranches
        return Optional.ofNullable(type).map(type ->
            switch (type) {
                case "ClassificationScorer" -> new ClassificationScorer(network);
                default -> throw new IllegalArgumentException("Unknown scorer type: " + type);
            }
        );
    }

    public Optional<LossFunction> getLossFunction() {
        if (lossFunction == null) {
            return Optional.empty();
        }
        return Optional.of(Config.getStaticField(LossFunction.class, lossFunction));
    }

}
