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

/**
 * Ausführbare Konfiguration mithilfe einer TOML-Datei.
 */
@Data
public class Config implements TomlSerializable {

    /**
     * Objekte dieser Klasse sollen nicht manuell instanziiert werden (deshalb package private)
     */
    Config() {
    }

    private ConfData data;
    private ConfNetwork network;
    private ConfTrainer trainer;
    private ConfScorer scorer;
    private ConfVisualization visualization;

    /**
     * Führt das Programm (Daten einlesen, Netzwerk erzeugen/trainieren/evaluieren) entsprechend der Konfiguration aus.
     *
     * @throws IOException Wird geworfen, wenn beim Einlesen der Daten oder bei Operationen auf dem Netzwerk ein Fehler auftritt.
     */
    public void execute() throws IOException {
        TrainTestSplit data = getData() != null ? getData().create() : new TrainTestSplit(null, null);
        Network network = getNetwork().create(data.train);

        long trainStart = 0;
        long trainStop = 0;
        if (getTrainer() != null) {
            Trainer trainer = getTrainer().create(network);
            if (getVisualization() != null && getVisualization().isEnabled()) {
                trainer.setViewManager(getVisualization().create(network));
            }

            trainStart = System.currentTimeMillis();
            trainer.train(data.train);
            trainStop = System.currentTimeMillis();

            getTrainer().export(network);
        } else if (getVisualization() != null && getVisualization().isEnabled()) {
            getVisualization().create(network).nextEpoch(1, network, 1);
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

    /**
     * Liest eine TOML-Konfigurationsdatei und erstellt aus dieser ein Config-Objekt
     *
     * @param filePath Dateipfad zur TOML-Config
     * @return Config-Objekt, das mit {@link Config#execute()} ausgeführt werden kann
     */
    public static Config read(final String filePath) {
        JToml jtoml = JToml.jToml();
        TomlTable rawConf = jtoml.read(Path.of(filePath)).asTable();
        return jtoml.fromToml(Config.class, rawConf);
    }

    /**
     * Ruft den Wert eines statischen Feldes aus einer angegebenen Klasse ab.
     *
     * @param <T>       Der erwartete Typ des Werts des statischen Felds.
     * @param clazz     Die Klasse, aus der das statische Feld abgerufen werden soll.
     * @param fieldName Der Name des statischen Feldes, auf das zugegriffen werden soll.
     * @return Der Wert des statischen Feldes, in den angegebenen Typ umgewandelt.
     */
    @SuppressWarnings("unchecked")
    @SneakyThrows
    public static <T> T getStaticField(Class<T> clazz, String fieldName) {
        return (T) clazz.getField(fieldName).get(null);
    }

}
