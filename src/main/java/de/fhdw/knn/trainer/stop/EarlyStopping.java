package de.fhdw.knn.trainer.stop;

import java.util.LinkedList;
import java.util.List;

/**
 * Implementierung einer StopFunction, die das Training dann beendet,
 * wenn der Fortschritt (Minimierung des Loss) zu gering wird
 */
public class EarlyStopping implements StopFunction {

    public final double minDelta;
    public final int patience;
    public final List<Double> losses = new LinkedList<>(List.of(Double.MAX_VALUE));

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

    private int epochsWaited = 0;

    @Override
    public boolean isFinished(double loss) {
        if (losses.getLast() - loss < minDelta) {
            epochsWaited += 1;
            losses.add(loss);
            return epochsWaited >= patience;
        } else {
            epochsWaited = 0;
            losses.add(loss);
            return false;
        }
    }

}
