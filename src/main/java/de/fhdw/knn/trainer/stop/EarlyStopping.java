package de.fhdw.knn.trainer.stop;

import io.github.wasabithumb.jtoml.serial.TomlSerializable;

/**
 * Implementierung einer StopFunction, die das Training dann beendet,
 * wenn der Fortschritt (Minimierung des Loss) zu gering wird
 */
public class EarlyStopping implements StopFunction, TomlSerializable {

    /**
     * Minimale Verbesserung des Verlusts seit der letzten Epoche
     */
    private final double minDelta;
    /**
     * Anzahl aufeinanderfolgender Epochen, in denen die Loss-Verbesserung minDelta unterschreiten muss
     */
    private final int patience;
    /**
     * Zuletzt berechneter Verlust zum Vergleich
     */
    private double previousLoss = Double.MAX_VALUE;
    /**
     * Anzahl bereits gewarteter Epochen ohne ausreichende Verbesserung des Verlusts
     */
    private int epochsWaited = 0;

    /**
     * Beendet das Training, wenn sich der Loss für ausreichend viele Epochen nicht ausreichend verbessert hat
     *
     * @param minDelta minimaler Fortschritt (minimal notwendige Verbesserung des Loss)
     * @param patience Anzahl aufeinanderfolgender Epochen, in denen die Loss-Verbesserung minDelta unterschreiten muss
     * @see EarlyStopping
     */
    public EarlyStopping(double minDelta, int patience) {
        this.minDelta = minDelta;
        this.patience = patience;
    }

    @Override
    public boolean isFinished(double loss) {
        if (previousLoss - loss < minDelta) {
            epochsWaited += 1;
            previousLoss = loss;
            return epochsWaited >= patience;
        } else {
            epochsWaited = 0;
            previousLoss = loss;
            return false;
        }
    }

}
