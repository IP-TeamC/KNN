package de.fhdw.knn.visualization;

import de.fhdw.knn.network.Network;
import javafx.application.Platform;

/**
 * Die Klasse {@code ViewManager} ist für die Verwaltung und Aktualisierung grafischer Ansichten
 * wie Heatmaps und Sankeyplots während des Trainingsprozesses eines neuronalen Netzwerks zuständig.
 * Sie übernimmt die Initialisierung und regelmäßige Aktualisierung dieser Ansichten in festgelegten Intervallen.
 */
public class ViewManager {

    /**
     * Stellt eine grafische Komponente dar, die für die Anzeige von Heatmap-Visualisierungen
     * der Gewichtsmatrizen eines neuronalen Netzwerks verantwortlich ist. Die Heatmap-Ansicht bietet
     * interaktive, scrollbare und zoombare Diagramme und unterstützt das Hinzufügen mehrerer Tabs für Visualisierungen.
     *
     * <p>Die Heatmap-Visualisierungskomponente. Wird initialisiert, wenn {@code heatmapConfig}
     * nicht {@code null} und das Intervall ungleich {@code 0} ist.
     *
     * @see ViewManager
     */
    private HeatmapView heatmapView;
    /**
     * Stellt die Sankey-Visualisierungskomponente dar, die zur Anzeige von Netzwerkdaten
     * in einem grafischen Sankey-Diagramm verwendet wird. Dieses Feld ist eine Instanz von {@code SankeyView},
     * die Methoden zur Initialisierung, zum Rendern und zur Aktualisierung des Sankey-Plots basierend
     * auf den Netzwerkdaten und dem Epochenfortschritt bereitstellt.
     *
     * <p>Die Sankey-Visualisierungskomponente. Wird initialisiert, wenn {@code sankeyConfig}
     * nicht {@code null} und das Intervall ungleich {@code 0} ist.
     *
     * @see ViewManager
     */
    private SankeyView sankeyView;

    /**
     * Die Konfiguration für die Heatmap-Visualisierung.
     *
     * @see HeatmapConfig
     */
    private final HeatmapConfig heatmapConfig;
    /**
     * Die Konfiguration für die Sankey-Visualisierung.
     *
     * @see SankeyConfig
     */
    private final SankeyConfig sankeyConfig;

    /**
     * Initialisiert den {@code ViewManager} mit angegebenen Intervallen für die Aktualisierung der Heatmap und dem Sankeyplot.
     * Wenn eine Konfiguration nicht {@code null} und das Intervall ungleich {@code 0} ist,
     * wird die entsprechende Visualisierung initialisiert.
     *
     * @param heatmapConfig Konfiguration der Heatmap-Visualisierung. Darf {@code null} sein.
     * @param sankeyConfig  Konfiguration der Sankey-Visualisierung. Darf {@code null} sein.
     * @param initNetwork   Die Netzwerkdaten zur Initialisierung der Diagramme. Darf {@code null} sein.
     * @see HeatmapConfig
     * @see SankeyConfig
     */
    public ViewManager(HeatmapConfig heatmapConfig, SankeyConfig sankeyConfig, Network initNetwork) {
        this.heatmapConfig = heatmapConfig;
        this.sankeyConfig = sankeyConfig;

        if (heatmapConfig != null && heatmapConfig.interval() > 0) {
            this.heatmapView = new HeatmapView();

            heatmapView.setThreshold(heatmapConfig.threshold());
            heatmapView.setWeightFilter(heatmapConfig.weightFilter());
            heatmapView.setShowWeights(heatmapConfig.showWeights());
            heatmapView.setNormalizeColors(heatmapConfig.normalizeColors());
            heatmapView.setColorScheme(heatmapConfig.colorScheme());

            if (initNetwork != null) {
                this.heatmapView.addHeatmap(new HeatmapData(initNetwork), "Epoche 0");
            }
        }

        if (sankeyConfig != null && sankeyConfig.interval() > 0) {
            if (initNetwork != null) {
                getOrCreateSankeyView(initNetwork, "Epoche 0");
            }
        }
    }

    /**
     * Aktualisiert die Heatmap und den Sankeyplot basierend auf der aktuellen Epoche,
     * den konfigurierten Intervallen und der Gesamtzahl der Epochen.
     * Bei {@code interval == -1} wird die {@code HeatmapView} erst bei der letzten Epoche erstellt und geöffnet.
     *
     * @param epoch     Die aktuelle Epoche.
     * @param network   Die für Aktualisierungen zu verwendenden Daten des neuronalen Netzwerks.
     * @param maxEpochs Die maximale Anzahl von Epochen im Trainingsprozess.
     */
    public void nextEpoch(final int epoch, Network network, int maxEpochs) {
        // Heatmap Update
        if (heatmapConfig != null && shouldUpdate(epoch, heatmapConfig.interval(), maxEpochs)) {
            getOrCreateHeatmapView().addHeatmap(new HeatmapData(network), "Epoche " + epoch);
        }

        // Sankey Update
        if (sankeyConfig != null && shouldUpdate(epoch, sankeyConfig.interval(), maxEpochs)) {
            String label = "Epoche " + epoch;
            if (sankeyView == null) {
                sankeyView = getOrCreateSankeyView(network, label);
            } else {
                sankeyView.update(new SankeyData(network), label);
            }
        }
    }

    /**
     * Erzwingt eine Aktualisierung der Heatmap bzw. des Sankeyplots bei einem EarlyStop.
     * Falls die {@code HeatmapView} noch nicht initialisiert wurde (z. B. bei {@code interval == -1}),
     * wird sie hier lazy erstellt und geöffnet.
     *
     * <p>Diese Methode wird in der Regel aufgerufen, wenn der Trainingsprozess vorzeitig durch
     * {@link de.fhdw.knn.trainer.stop.EarlyStopping} abgebrochen wird,
     * um den Endzustand in den Visualisierungen darzustellen.
     *
     * @param epoch   Die aktuelle Epoche.
     * @param network Die für Aktualisierungen zu verwendenden Daten des neuronalen Netzwerks.
     */
    public void earlyStop(int epoch, Network network) {
        String label = "Epoche " + epoch + " (Early Stopping)";
        if (heatmapConfig != null && heatmapConfig.interval() != 0) {
            getOrCreateHeatmapView().addHeatmap(
                    new HeatmapData(network), label);
        }
        if (sankeyConfig != null && sankeyConfig.interval() != 0) {
            getOrCreateSankeyView(network, label).update(new SankeyData(network), label);
        }
    }

    /**
     * Evaluiert, ob für eine bestimmte Epoche eine Aktualisierung ausgelöst werden soll.
     *
     * <p>Die Methode evaluiert drei Bedingungen:
     *
     * <br>1. Ob es sich um die erste Epoche handelt.
     * <br>2. Ob die Epoche ein Vielfaches des aktuellen Aktualisierungsintervalls ist.
     * <br>3. Ob es sich um die letzte Epoche handelt.
     *
     * <p>Bei einem Intervall von {@code -1} wird ausschließlich die letzte Epoche aktualisiert.
     *
     * @param epoch     Die aktuelle Epochennummer.
     * @param interval  Das Intervall, in welchem eine Aktualisierung auftreten soll.
     *                  {@code -1} bedeutet nur am Trainingsende.
     * @param maxEpochs Die maximale Anzahl der Epochen.
     * @return {@code true} wenn eine Aktualisierung für diese Epoche ausgelöst werden soll,
     * {@code false} andernfalls.
     */
    private boolean shouldUpdate(int epoch, int interval, int maxEpochs) {
        if (interval == -1) {
            return epoch == maxEpochs;
        }
        if (interval == 0) {
            return false;
        }
        return epoch == 1 || epoch % interval == 0 || epoch == maxEpochs;
    }

    /**
     * Gibt die bestehende {@code HeatmapView} zurück oder erstellt sie lazy,
     * falls sie noch nicht initialisiert wurde.
     *
     * @return Die initialisierte {@code HeatmapView}-Instanz.
     */
    private HeatmapView getOrCreateHeatmapView() {
        if (heatmapView == null) {
            heatmapView = new HeatmapView();
            heatmapView.setThreshold(heatmapConfig.threshold());
            heatmapView.setWeightFilter(heatmapConfig.weightFilter());
            heatmapView.setShowWeights(heatmapConfig.showWeights());
            heatmapView.setNormalizeColors(heatmapConfig.normalizeColors());
            heatmapView.setColorScheme(heatmapConfig.colorScheme());
        }
        return heatmapView;
    }

    /**
     * Gibt die bestehende {@code SankeyView} zurück oder erstellt sie lazy,
     * falls sie noch nicht initialisiert wurde.
     *
     * @param network Die Netzwerkdaten für die initiale Darstellung.
     * @param label   Der Anzeigetitel des ersten Tabs.
     * @return Die initialisierte {@code SankeyView}-Instanz.
     */
    private SankeyView getOrCreateSankeyView(Network network, String label) {
        if (sankeyView == null) {
            try {
                Platform.startup(() -> {
                });
            } catch (IllegalStateException ignored) {
                // Ignorieren, falls JavaFX bereits läuft
            }
            sankeyView = new SankeyView(new SankeyData(network), label, sankeyConfig.threshold(), sankeyConfig.weightFilter());
        }
        return sankeyView;
    }
}
