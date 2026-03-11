package de.fhdw.knn.config.trainer;

import de.fhdw.knn.config.Config;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

import java.util.Optional;

/**
 * Die Klasse {@code ConfTrainer} stellt eine Konfiguration für einen {@code Trainer} in einem neuronalen Netzwerk dar.
 *
 * <p>Diese Klasse wird verwendet, um einen Trainer mit spezifischen Konfigurationen wie Verlustfunktion,
 * Bedingungen für EarlyStopping, Optimierungsparametern und Exporteinstellungen zu erstellen.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 *
 * @see Trainer
 * @see TomlSerializable
 */
@Data
public class ConfTrainer implements TomlSerializable {

    /**
     * Stellt die Verlustfunktion dar, die während des Trainings verwendet werden soll.
     *
     * @see ConfTrainer#create(Network)
     * @see LossFunction
     */
    private String lossFunction;
    /**
     * Konfiguriert die Bedingungen für EarlyStopping während des Trainings.
     *
     * @see ConfTrainer#create(Network)
     */
    private ConfEarlyStopping earlyStopping;

    /**
     * Gibt die maximale Anzahl von Epochen für den Trainingsprozess an.
     *
     * @see ConfTrainer#create(Network)
     */
    private int maxEpochs;
    /**
     * Gibt an, ob die Trainingsdaten vor jeder Epoche neu gemischt werden sollen.
     * Empfohlen für eine bessere Generalisierung.
     *
     * @see ConfTrainer#create(Network)
     */
    private boolean shuffleEpoch;

    /**
     * Gibt an, wie viele Trainingsbeispiele gleichzeitig verarbeitet werden, bevor die Gewichte des Netzwerks angepasst werden.
     *
     * @see ConfTrainer#create(Network)
     */
    private int batchSize;
    /**
     * Gibt die Lernrate des Netzwerks an.
     *
     * @see ConfTrainer#create(Network)
     */
    private double learningRate;

    /**
     * Gibt den Dateipfad an, in den das trainierte neuronale Netzwerk exportiert werden soll.
     *
     * @see ConfTrainer#export(Network)
     */
    private String exportFile;

    /**
     * Erstellt eine neue {@code Trainer}-Instanz, die mit den Parametern dieses {@code ConfTrainer} konfiguriert ist.
     *
     * @param network das zu trainierende neuronale Netzwerk.
     * @return eine {@code Trainer}-Instanz, die mit der angegebenen Verlustfunktion, den Kriterien für
     * das vorzeitige Beenden, der Optimierungsfunktion und anderen Parametern
     * für das Training des bereitgestellten Netzwerks konfiguriert ist.
     * @see Trainer
     */
    public Trainer create(Network network) {
        LossFunction lossFunction = Config.getStaticField(LossFunction.class, this.lossFunction);
        StopFunction stopFunction = Optional.ofNullable(earlyStopping)
                .map(ConfEarlyStopping::create)
                .orElse(StopFunction.NEVER);
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(learningRate));
        return new Trainer(network, maxEpochs, shuffleEpoch, batchSize, lossFunction, stopFunction, optimizationFunction);
    }

    /**
     * Exportiert das trainierte neuronale Netzwerk in den angegebenen Dateipfad.
     *
     * @param network das zu exportierende neuronale Netzwerk.
     */
    public void export(Network network) {
        if (exportFile != null) {
            network.export(exportFile);
        }
    }
}
