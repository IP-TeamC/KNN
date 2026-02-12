package de.fhdw.knn.config.trainer;

import de.fhdw.knn.config.Config;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

import java.util.Optional;

@Data
public class ConfTrainer implements TomlSerializable {

    private String lossFunction;
    private EarlyStopping earlyStopping;

    private int maxEpochs;
    private boolean shuffleEpoch;
    private int batchSize;
    private double learningRate;
    private boolean visualization;

    public Trainer create(Network network) {
        LossFunction lossFunction = Config.getStaticField(LossFunction.class, this.lossFunction);
        StopFunction stopFunction = Optional.ofNullable(earlyStopping)
                .map(value -> (StopFunction) value)
                .orElse(StopFunction.NEVER);
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(learningRate));
        return new Trainer(network, maxEpochs, shuffleEpoch, batchSize, lossFunction, stopFunction, optimizationFunction, visualization);
    }

}
