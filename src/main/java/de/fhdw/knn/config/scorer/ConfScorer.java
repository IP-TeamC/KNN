package de.fhdw.knn.config.scorer;

import de.fhdw.knn.config.Config;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.scorer.ClassificationScorer;
import de.fhdw.knn.scorer.Scorer;
import de.fhdw.knn.trainer.loss.LossFunction;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;
import lombok.Data;

import java.util.Optional;

/**
 * Die Klasse {@code ConfScorer} stellt eine Konfiguration für einen {@code Scorer} in einem neuronalen Netzwerk dar.
 *
 * <p>Diese Klasse unterstützt die dynamische Instanziierung eines Scorers und
 * das Abrufen eines Verlustfunktionstyps aus der Konfiguration.
 *
 * <p>Implementiert die Schnittstelle {@code TomlSerializable}, um die TOML-basierte Serialisierung zu ermöglichen.
 */
@Data
public class ConfScorer implements TomlSerializable {

    /**
     * Stellt die Verlustfunktion dar, die für die Evaluierung des neuronalen Netzwerks verwendet wird.
     *
     * @see ConfScorer#create(Network)
     * @see LossFunction
     */
    private String lossFunction;
    /**
     * Stellt den Typ des Scorers dar, der in der Konfiguration angegeben wird.
     *
     * @see ConfScorer#create(Network)
     * @see Scorer
     */
    private String type;

    /**
     * Erstellt dynamisch eine Instanz von {@code Scorer} basierend auf dem Konfigurationstyp.
     *
     * @param network das neuronale Netzwerk, das vom {@code Scorer} für die Bewertung verwendet wird.
     * @return ein {@code Optional}, das den erstellten {@code Scorer} enthält, wenn der Typ erkannt wird;
     *         andernfalls ein leeres {@code Optional}.
     * @throws IllegalArgumentException wenn der angegebene Typ unbekannt ist.
     *
     * @see Scorer
     */
    public Optional<Scorer> create(Network network) {
        //noinspection SwitchStatementWithTooFewBranches
        return Optional.ofNullable(type).map(type ->
            switch (type) {
                case "ClassificationScorer" -> new ClassificationScorer(network);
                default -> throw new IllegalArgumentException("Unknown scorer type: " + type);
            }
        );
    }

    /**
     * Ruft die für den {@code Scorer} konfigurierte Verlustfunktion ab.
     *
     * @return ein {@code Optional}, das die aufgelöste {@code LossFunction} enthält, falls verfügbar;
     *         andernfalls ein leeres {@code Optional}.
     *
     * @see LossFunction
     */
    public Optional<LossFunction> getLossFunction() {
        if (lossFunction == null) {
            return Optional.empty();
        }
        return Optional.of(Config.getStaticField(LossFunction.class, lossFunction));
    }
}
