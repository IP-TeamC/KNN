package de.fhdw.knn.config.visualization;

import de.fhdw.knn.visualization.SankeyConfig;
import de.fhdw.knn.visualization.WeightFilter;
import io.github.wasabithumb.jtoml.serial.TomlSerializable;

/**
 * Konfiguration für die Sankey-Visualisierung.
 *
 * <p>Wird als verschachtelte Sektion {@code [visu.sankey]} in der TOML-Konfiguration verwendet.
 *
 * @see ConfVisualization
 * @see TomlSerializable
 */
public class ConfSankey implements TomlSerializable {

    /**
     * Objekte dieser Klasse sollen nicht manuell instanziiert werden (deshalb package private).
     */
    ConfSankey() {
    }

    /**
     * Stellt das Intervall (in Epochen) dar, in dem das Sankeyplot aktualisiert wird.
     *
     * <p>Ein Wert größer als 0 gibt die Periodizität der Aktualisierungen an (z. B. alle n Epochen).
     * Der Wert {@code -1} aktiviert das Sankeyplot ausschließlich am Ende des Trainings.
     * Der Wert {@code 0} deaktiviert das Sankeyplot vollständig.
     */
    public int interval = 0;

    /**
     * Schwellenwert für die Visualisierung von Gewichtsverbindungen.
     * Verbindungen mit {@code |weight| ≤ threshold} werden nicht dargestellt.
     *
     * <p>Gilt für Heatmap und Sankey.
     */
    public double threshold = 0.1;

    /**
     * Gibt an, ob positive, negative oder alle Gewichte visualisiert werden sollen.
     * Mögliche Werte: {@code POSITIVE}, {@code NEGATIVE}, {@code BOTH}.
     *
     * <p>Gilt für Heatmap und Sankey.
     *
     * @see de.fhdw.knn.visualization.WeightFilter
     */
    public String weightFilter = "BOTH";

    /**
     * Konvertiert diese TOML-Konfiguration in ein {@link SankeyConfig}-Record.
     *
     * @return eine unveränderliche {@link SankeyConfig}-Instanz mit den aktuellen Konfigurationswerten.
     * @throws IllegalArgumentException wenn {@code weightFilter} keinen gültigen Enum-Wert entspricht.
     * @see SankeyConfig
     */
    public SankeyConfig toConfig() {
        return new SankeyConfig(
                interval,
                threshold,
                WeightFilter.valueOf(weightFilter.toUpperCase())
        );
    }
}