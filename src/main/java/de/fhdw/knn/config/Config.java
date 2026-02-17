package de.fhdw.knn.config;

import de.fhdw.knn.config.data.ConfData;
import de.fhdw.knn.config.network.ConfNetwork;
import de.fhdw.knn.config.scorer.ConfScorer;
import de.fhdw.knn.config.trainer.ConfTrainer;
import de.fhdw.knn.config.visualization.ConfVisualization;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.scorer.Score;
import de.fhdw.knn.scorer.Scorer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.loss.LossFunction;
import io.github.wasabithumb.jtoml.JToml;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import io.github.wasabithumb.jtoml.value.table.TomlTable;
import lombok.Data;
import lombok.SneakyThrows;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

@Data
public class Config implements TomlSerializable {

    private ConfData data;
    private ConfNetwork network;
    private ConfTrainer trainer;
    private ConfScorer scorer;
    private ConfVisualization visualization;

    public void execute() throws IOException {
        TrainTestSplit data = getData().create();
        Network network = getNetwork().create(data.train);

        long trainStart = 0;
        long trainStop = 0;
        if (getTrainer() != null) {
            Trainer trainer = getTrainer().create(network);

            trainStart = System.currentTimeMillis();
            trainer.train(data.train);
            trainStop = System.currentTimeMillis();

            getTrainer().export(network);
        }

        long testStart = 0;
        long testStop = 0;
        if (getScorer() != null) {
            Optional<LossFunction> lossFunction = getScorer().getLossFunction();
            Optional<Scorer> scorer = Optional.of(getScorer())
                    .flatMap(conf -> conf.create(network));

            testStart = System.currentTimeMillis();
            Optional<Score> score = scorer.map(scorerLocal -> scorerLocal.score(data.test));
            double[][] testPredictions = network.predict(data.test.inputs);
            Optional<Double> testLoss = lossFunction.map(scorerLocal -> scorerLocal.totalLoss(data.test.outputs, testPredictions));
            testStop = System.currentTimeMillis();

            System.out.println();
            score.ifPresent(Score::print);
            testLoss.ifPresent(testLossLocal -> System.out.println("Test Loss: " + testLossLocal));
        }

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.println();
        if (trainStart > 0) System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        if (testStart > 0) System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);
    }

    public static Config read(final String filePath) {
        JToml jtoml = JToml.jToml();
        TomlTable rawConf = jtoml.read(Path.of(filePath)).asTable();
        return jtoml.fromToml(Config.class, rawConf);
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public static <T> T getStaticField(Class<T> clazz, String fieldName) {
        return (T) clazz.getField(fieldName).get(null);
    }

}
