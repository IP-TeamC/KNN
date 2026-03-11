package de.fhdw.knn.config.trainer;

import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

/**
 * Die Klasse {@code ConfEarlyStopping} stellt eine Konfiguration für die {@code StopFunction} {@code EarlyStopping} des Trainings eines neuronalen Netzwerks dar.
 *
 * <p>Diese Klasse wird verwendet, um eine StopFunction mit Bedingungen für das EarlyStopping zu erstellen.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 *
 * @see EarlyStopping
 * @see TomlSerializable
 */
@Data
public class ConfEarlyStopping implements TomlSerializable {

    /**
     * Objekte dieser Klasse sollen nicht manuell instanziiert werden (deshalb package private)
     */
    ConfEarlyStopping() {
    }

    /**
     * Stellt die minimale Verbesserung des Verlusts seit der letzten Epoche ein
     *
     * @see ConfEarlyStopping#create()
     * @see EarlyStopping
     */
    private double minDelta;
    /**
     * Konfiguriert die Anzahl aufeinanderfolgender Epochen, in denen die Loss-Verbesserung minDelta unterschreiten muss
     *
     * @see ConfEarlyStopping#create()
     * @see EarlyStopping
     */
    private int patience;

    /**
     * Erstellt eine neue {@code EarlyStopping}-Instanz, die mit den Parametern dieses {@code ConfEarlyStopping} konfiguriert ist.
     *
     * @return eine {@code EarlyStopping}-Instanz, die mit den angegebenen Kriterien für das vorzeitige Beenden konfiguriert ist.
     * @see Trainer
     */
    public StopFunction create() {
        return new EarlyStopping(minDelta, patience);
    }

}
