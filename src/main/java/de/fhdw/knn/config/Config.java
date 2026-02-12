package de.fhdw.knn.config;

import de.fhdw.knn.config.data.ConfData;
import de.fhdw.knn.config.network.ConfNetwork;
import de.fhdw.knn.config.scorer.ConfScorer;
import de.fhdw.knn.config.trainer.ConfTrainer;
import de.fhdw.knn.config.visualization.ConfVisualization;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.scorer.Score;
import de.fhdw.knn.trainer.Trainer;
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
        Trainer trainer = getTrainer().create(network);

        long trainStart = System.currentTimeMillis();
        trainer.train(data.train);
        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        Optional<Score> score = Optional.ofNullable(getScorer())
                .flatMap(conf -> conf.create(network))
                .map(scorer -> scorer.score(data.test));
        double[][] testPredictions = network.predict(data.test.inputs);
        double testLoss = trainer.lossFunction.totalLoss(data.test.outputs, testPredictions);
        long testStop = System.currentTimeMillis();
        System.out.println();
        score.ifPresent(Score::print);
        System.out.println("Test Loss: " + testLoss);

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("\nTrain Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);
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
